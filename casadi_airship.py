import casadi as ca
import numpy as np
from airship_dynamic import AirshipCasADiSymbolic

# Simulation parameters
T = 5.0      # Total time (s)
N = 50     # Number of time steps
dt = T / N    # Time step
xf = 5  # Final x position (m)

# Airship parameters
m = 2934 # Mass of the airship (kg)
g = 9.74  # Gravitational acceleration (m/s^2)
Ix = 393187
Iy = 1224880
Iz = 939666
Ixz = -62882
I0 = np.diag([Ix, Iy, Iz])


# Create symbolic variables for each time step
x_list, y_list, z_list, phi_list, theta_list, psi_list = [], [], [], [], [], []
vx_list, vy_list, vz_list, omega_phi_list, omega_theta_list, omega_psi_list = [], [], [], [], [], [] 
umag_list, ul_list, ur_list = [], [], []

for i in range(N):
    x_list.append(ca.SX.sym(f'x_{i}'))
    y_list.append(ca.SX.sym(f'y_{i}'))
    z_list.append(ca.SX.sym(f'z_{i}'))
    phi_list.append(ca.SX.sym(f'phi_{i}'))
    theta_list.append(ca.SX.sym(f'theta_{i}'))
    psi_list.append(ca.SX.sym(f'psi_{i}'))
    vx_list.append(ca.SX.sym(f'vx_{i}'))
    vy_list.append(ca.SX.sym(f'vy_{i}'))
    vz_list.append(ca.SX.sym(f'vz_{i}'))
    omega_phi_list.append(ca.SX.sym(f'omega_phi_{i}'))
    omega_theta_list.append(ca.SX.sym(f'omega_theta_{i}'))
    omega_psi_list.append(ca.SX.sym(f'omega_psi_{i}'))
    umag_list.append(ca.SX.sym(f'umag_{i}'))
    ul_list.append(ca.SX.sym(f'ul_{i}'))
    ur_list.append(ca.SX.sym(f'ur_{i}'))
    
# Stack all variables into a single optimization vector
X = ca.vertcat(*x_list, *y_list, *z_list, *phi_list, *theta_list, *psi_list,
               *vx_list, *vy_list, *vz_list, *omega_phi_list, *omega_theta_list, 
               *omega_psi_list, *umag_list, *ul_list, *ur_list)

# Helpers to extract state/control at a given timestep
def get_state(i):
    return ca.vertcat(x_list[i], y_list[i], z_list[i], 
                      phi_list[i], theta_list[i], psi_list[i],
                      vx_list[i], vy_list[i], vz_list[i],
                      omega_phi_list[i], omega_theta_list[i], omega_psi_list[i])

def get_control(i):
    return ca.vertcat(umag_list[i], ul_list[i], ur_list[i])

# Dynamics function
dynamics = AirshipCasADiSymbolic()

# Dynamics and boundary constraints
g = []

for i in range(N - 1):
    s_i = get_state(i)
    s_ip1 = get_state(i + 1)
    u_i = get_control(i)
    u_ip1 = get_control(i + 1)
    f_i = dynamics.rhs_symbolic(s_i, u_i)
    f_ip1 = dynamics.rhs_symbolic(s_ip1, u_ip1)
    g.append(s_ip1 - s_i - (dt/2)*(f_i + f_ip1))   

# Initial and final conditions
g.append(get_state(0) - ca.vertcat(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))  # Start at rest at origin
g.append(get_state(N - 1) - ca.vertcat(xf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))  # End at position (100, 0, 0) with no velocity or rotation 
    
# Objective: minimize total control effort
f = 0
for i in range(N - 1):
    u_i = get_control(i)
    u_ip1 = get_control(i + 1)
    f += 0.5 * dt * (ca.mtimes(u_i.T, u_i) + ca.mtimes(u_ip1.T, u_ip1))
    
# Stack all constraints
G = ca.vertcat(*g)

# Define the optimization problem
nlp = {'x': X, 'f': f, 'g': G}

# Create an NLP solver
solver = ca.nlpsol('solver', 'ipopt', nlp,{'ipopt': {'max_iter': 10000}})

# Bound on variables
v_max = 50 # Maximum velocity (m/s)
theta_max = np.pi/12 # Maximum pitch angle (radians)
T_max = 10000 # Maximum thrust (N)
lbx = [-ca.inf]*(3*N) + [-theta_max]*(3*N) + [-v_max]*(3*N) + [-ca.inf]*(3*N) + [-T_max]*N + [-ca.inf]*N + [-ca.inf]*N
ubx = [ca.inf]*(3*N) + [theta_max]*(3*N) + [v_max]*(3*N) + [ca.inf]*(3*N) + [T_max]*N + [ca.inf]*N + [ca.inf]*N
    
# Bounds on constraints
lbg = [0] * (N - 1) * 12 + [0] * 2 *12 # Equality constraints (g == 0)
ubg = [0] * (N - 1) * 12 + [0] * 2 *12 # Equality constraints (g == 0)

# Initial guess for the optimization variables
x0 = []
for i in range(N):
    x0 += [xf*i/N]
for i in range(N*14):
    x0 += [10]

# Solve the optimization problem
sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

if __name__ == "__main__":
    # Example usage of the dynamics function
    s = get_state(0)  # Get the state at the first time step
    u = get_control(0)  # Get the control at the first time step
    dynamics.rhs_symbolic(s, u)  # Print the dynamics output for the first state and control