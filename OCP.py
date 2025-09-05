import casadi as ca
import numpy as np
from airship_dynamic import AirshipCasADiSymbolic

class OCP_AS():
    def __init__(self, T=5.0, N=30, dynamic_model=None):
        
        # Simulation parameters
        self.T = T      # Total time (s)
        self.N = N     # Number of time steps
        self.dt = T / N    # Time step
        self.dynamics = dynamic_model if dynamic_model is not None else AirshipCasADiSymbolic() # Use provided dynamic model or default
        
        # Bound on variables
        self.v_max = 50 # Maximum velocity (m/s)
        self.theta_max = np.pi/6 # Maximum pitch angle (radians)
        self.T_max = 100000 # Maximum thrust (N)
        self.lr_max = np.pi/4 # Maximum roll angle (radians)


    def setup(self):
        return

        
    
    def get_state(self, i):
        return ca.vertcat(self.x_list[i], self.y_list[i], self.z_list[i], 
                          self.phi_list[i], self.theta_list[i], self.psi_list[i],
                          self.vx_list[i], self.vy_list[i], self.vz_list[i],
                          self.omega_phi_list[i], self.omega_theta_list[i], self.omega_psi_list[i])
        
        
    def get_control(self, i):
        return ca.vertcat(self.umag_list[i], self.ul_list[i], self.ur_list[i])
    
    
    def forward(self, state_target:np.array, state_init:np.array,R=ca.diag([1.0, 1.0, 1.0])):
        self.R = R
        # Create symbolic variables for each time step
        self.x_list, self.y_list, self.z_list, self.phi_list, self.theta_list, self.psi_list = [], [], [], [], [], []
        self.vx_list, self.vy_list, self.vz_list, self.omega_phi_list, self.omega_theta_list, self.omega_psi_list = [], [], [], [], [], [] 
        self.umag_list, self.ul_list, self.ur_list = [], [], []
        
        for i in range(self.N):
            self.x_list.append(ca.SX.sym(f'x_{i}'))
            self.y_list.append(ca.SX.sym(f'y_{i}'))
            self.z_list.append(ca.SX.sym(f'z_{i}'))
            self.phi_list.append(ca.SX.sym(f'phi_{i}'))     # roll (around x-axis)
            self.theta_list.append(ca.SX.sym(f'theta_{i}')) # pitch (around y-axis)
            self.psi_list.append(ca.SX.sym(f'psi_{i}'))     # yaw (around z-axis)
            self.vx_list.append(ca.SX.sym(f'vx_{i}'))
            self.vy_list.append(ca.SX.sym(f'vy_{i}'))
            self.vz_list.append(ca.SX.sym(f'vz_{i}'))
            self.omega_phi_list.append(ca.SX.sym(f'omega_phi_{i}'))
            self.omega_theta_list.append(ca.SX.sym(f'omega_theta_{i}'))
            self.omega_psi_list.append(ca.SX.sym(f'omega_psi_{i}'))
            self.umag_list.append(ca.SX.sym(f'umag_{i}'))
            self.ul_list.append(ca.SX.sym(f'ul_{i}'))
            self.ur_list.append(ca.SX.sym(f'ur_{i}'))
            
        # Stack all variables into a single optimization vector
        self.X = ca.vertcat(*self.x_list, *self.y_list, *self.z_list, *self.phi_list, *self.theta_list, *self.psi_list,
               *self.vx_list, *self.vy_list, *self.vz_list, *self.omega_phi_list, *self.omega_theta_list, 
               *self.omega_psi_list, *self.umag_list, *self.ul_list, *self.ur_list)
        
        self.g = []

        for i in range(self.N - 1):
            s_i = self.get_state(i)
            s_ip1 = self.get_state(i + 1)
            u_i = self.get_control(i)
            u_ip1 = self.get_control(i + 1)
            f_i = self.dynamics.rhs_symbolic(s_i, u_i)
            f_ip1 = self.dynamics.rhs_symbolic(s_ip1, u_ip1)
            self.g.append(s_ip1 - s_i - (self.dt/2)*(f_i + f_ip1))
            
        self.f = 0
        
        for i in range(self.N - 1):
            u_i = self.get_control(i)
            u_ip1 = self.get_control(i + 1)
            self.f += .5 * self.dt * (u_i.T@R@ u_i + u_ip1.T@R@u_ip1)
        N = self.N
        # Initial and final conditions
        start = [self.get_state(0) - ca.SX(state_init)]
        end   = [self.get_state(self.N - 1) - ca.SX(state_init)]
        
        # Stack all constraints
        G = ca.vertcat(*self.g, *start, *end)

        # Define the optimization problem
        nlp = {'x': self.X, 'f': self.f, 'g': G}

        # Create an NLP solver
    #     solver_options = {
    #     'ipopt.print_level': 0,
    #     'print_time': 0,  # Suppress CasADi's own timing information
    #     'ipopt.sb': 'yes' # Suppress Ipopt's banner
    # }
        solver = ca.nlpsol('solver', 'ipopt', nlp,{'ipopt': {'max_iter': 10000,'print_level':0,'sb': 'yes'},'print_time': 0})
        
        # Define bounds
        lbx = [-ca.inf]*(3*self.N) + [-ca.inf]*(3*self.N) + [-self.v_max]*(3*self.N) + [-ca.inf]*(3*self.N) + [-self.T_max]*self.N + [-self.lr_max]*self.N + [-self.lr_max]*self.N
        ubx = [ ca.inf]*(3*self.N) + [ ca.inf]*(3*self.N) + [ self.v_max]*(3*self.N) + [ ca.inf]*(3*self.N) + [ self.T_max]*self.N + [ self.lr_max]*self.N + [ self.lr_max]*self.N

        # Bounds on constraints
        lbg = [0] * (self.N - 1) * 12 + [0] * 2 *12 # Equality constraints (g == 0)
        ubg = [0] * (self.N - 1) * 12 + [0] * 2 *12 # Equality constraints (g == 0)

        # Initial guess for the optimization variables
        x0 = []
        for state in range(12):
            for step in range(self.N):
                temp = state_init[state]+(state_target[state]-state_init[state])*step/self.N + np.random.uniform(-0.01, 0.01)
                x0 += [temp]
        for i in range(self.N):
            x0 += [1000]
        for i in range(2*self.N):
            x0 += [0.0] 
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        solver_stats = solver.stats()
        print(f"{solver_stats['return_status']}",end=": ")
        
        x_opt = sol['x']
        x_vals = x_opt[0:N]
        y_vals = x_opt[N:2*N]
        z_vals = x_opt[2*N:3*N]
        phi_vals = x_opt[3*N:4*N]
        theta_vals = x_opt[4*N:5*N]
        psi_vals = x_opt[5*N:6*N]
        vx_vals = x_opt[6*N:7*N]
        vy_vals = x_opt[7*N:8*N]
        vz_vals = x_opt[8*N:9*N]
        omega_phi_vals = x_opt[9*N:10*N]
        omega_theta_vals = x_opt[10*N:11*N]
        omega_psi_vals = x_opt[11*N:12*N]
        umag_vals = x_opt[12*N:13*N]
        ul_vals = x_opt[13*N:14*N]
        ur_vals = x_opt[14*N:15*N]
        
        return ca.horzcat(x_vals, y_vals, z_vals, phi_vals, theta_vals, psi_vals,
                          vx_vals, vy_vals, vz_vals, omega_phi_vals, omega_theta_vals, omega_psi_vals), ca.horzcat(umag_vals, ul_vals, ur_vals)


if __name__ == "__main__":
    
    n=30
    
    # Define the simulator
    dynamics = AirshipCasADiSymbolic()
    state = ca.SX.sym('state', 12)  # [x, y, z, phi, theta, psi, vx, vy, vz, omega_phi, omega_theta, omega_psi]
    control = ca.SX.sym('control', 3)  # [umag, ul, ur]
    rhs = dynamics.rhs_symbolic(state, control)
    f = ca.Function('f', [state, control], [rhs])
    
    # Define the OCP
    ocp = OCP_AS(T=.50, N=10)
    # ocp.setup(R=ca.diag([1.0, 1.0, 1.0]))
    
    # Initial and target states
    state_init  = np.array([0., 0.  , -2000., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    state_target= np.array([50., 0., -2000., 0., 0., 0., 50., 0., 0., 0., 0., 0.])
    increment = (state_target - state_init)/n

    traj = []
    # Solve the OCP
    for i in range(n):
        state_traj, control_traj = ocp.forward(state_init+increment, state_init)
        dxdt =f(state_init, control_traj[0,:])
        state_init = state_init + np.array(dxdt).flatten()*ocp.dt
        print(control_traj[1,:])
        print(state_init)
        traj+= [state_init[:3]]
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')
    traj = np.array(traj)
    ax.plot(traj[:,0], traj[:,1], traj[:,2], label='Trajectory', linewidth=10)
    ax.scatter(0, 0., -2000, color='green', label='Start')
    ax.scatter(0,50, -2000, color='red', label='End')
    # ax.set_ylim(-50,550)
    # ax.set_xlim(-50,50)
    # ax.set_zlim(-2000-5,-2000+5)
    ax.legend()
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    plt.show()