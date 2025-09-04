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


    def setup(self,R=ca.diag([1.0, 1.0, 1.0])):
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
            self.f += 500 * self.dt * (u_i.T@R@ u_i + u_ip1.T@R@u_ip1)

        
    
    def get_state(self, i):
        return ca.vertcat(self.x_list[i], self.y_list[i], self.z_list[i], 
                          self.phi_list[i], self.theta_list[i], self.psi_list[i],
                          self.vx_list[i], self.vy_list[i], self.vz_list[i],
                          self.omega_phi_list[i], self.omega_theta_list[i], self.omega_psi_list[i])
        
        
    def get_control(self, i):
        return ca.vertcat(self.umag_list[i], self.ul_list[i], self.ur_list[i])
    
    
    def forward(self, state_target, state_init):
        # Initial and final conditions
        start = [self.get_state(0) - state_init]
        end   = [self.get_state(self.N - 1) - state_init]
        
        # Stack all constraints
        G = ca.vertcat(*self.g, *start, *end)

        # Define the optimization problem
        nlp = {'x': self.X, 'f': self.f, 'g': G}

        # Create an NLP solver
        solver = ca.nlpsol('solver', 'ipopt', nlp,{'ipopt': {'max_iter': 10000}})
        
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
                x0 += [state_init[state]+(state_target[state]-state_init[state])*step/self.N + np.random.uniform(-0.01, 0.01)]
        for i in range(self.N):
            x0 += [1000]
        for i in range(2*self.N):
            x0 += [0.0] 
            
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        