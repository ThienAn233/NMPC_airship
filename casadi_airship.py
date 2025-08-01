import casadi as ca
import numpy as np
from airship_dynamic import AirshipCasADiSymbolic

# Simulation parameters
T = 5.0      # Total time (s)
N = 30     # Number of time steps
dt = T / N    # Time step
xf = 5  # Final x position (m)


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

def get_ref_traj(i):      
    return ca.vertcat(xf * i / (N - 1), 0, 0)

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
g.append(get_state(0) - ca.vertcat(0, 0, 0, 0., 0., np.pi, 0., 0., 0., 0., 0., 0.))  # Start at rest at origin
g.append(get_state(N - 1) - ca.vertcat(xf, 0, 0, 0., 0., np.pi, 0., 0., 0., 0., 0., 0.))  # End at position (100, 0, 0) with no velocity or rotation 
g.append(get_control(0) - ca.vertcat(0., 0., 0.))  # Initial control inputs
g.append(get_control(N - 1) - ca.vertcat(0., 0., 0.))  # Final control inputs

for i in range(N - 1):
    g.append(get_state(i)[1:3]-ca.vertcat(0,0))
    
# Objective: minimize total control effort
R = ca.diag([1.0, 1.0, 1.0])  # Control effort weighting matrix
f = 0
for i in range(N - 1):
    u_i = get_control(i)
    u_ip1 = get_control(i + 1)
    # xyz_i = get_state(i)[1:5]
    # xyz_ip1 = get_state(i + 1)[1:5]
    # xyz_ref_i = get_ref_traj(i)
    # xyz_ref_ip1 = get_ref_traj(i + 1)
    f += .5 * dt * (u_i.T@R@ u_i + u_ip1.T@R@u_ip1)
    # f += 50000 * dt * ((xyz_i).T @ (xyz_i) + (xyz_ip1).T @ (xyz_ip1))

    
# Stack all constraints
G = ca.vertcat(*g)

# Define the optimization problem
nlp = {'x': X, 'f': f, 'g': G}

# Create an NLP solver
solver = ca.nlpsol('solver', 'ipopt', nlp,{'ipopt': {'max_iter': 10000}})

# Bound on variables
v_max = 2 # Maximum velocity (m/s)
theta_max = np.pi/6 # Maximum pitch angle (radians)
T_max = 100000 # Maximum thrust (N)
lr_max = np.pi/4 # Maximum roll angle (radians)
lbx = [-ca.inf]*(3*N) + [-ca.inf]*(2*N) + [-ca.inf]*(N) + [-ca.inf]*(3*N) + [-ca.inf]*(3*N) + [-T_max]*N + [-lr_max]*N + [-lr_max]*N
ubx = [ ca.inf]*(3*N) + [ ca.inf]*(2*N) + [ ca.inf]*(N) + [ ca.inf]*(3*N) + [ ca.inf]*(3*N) + [ T_max]*N + [ lr_max]*N + [ lr_max]*N

# Bounds on constraints
lbg = [0] * (N - 1) * 12 + [-1e-6] * 2 *15 + [-.1] * 2 * (N-1) # Equality constraints (g == 0)
ubg = [0] * (N - 1) * 12 + [ 1e-6] * 2 *15 + [ .1] * 2 * (N-1)# Equality constraints (g == 0)

# Initial guess for the optimization variables
x0 = []
for i in range(N):
    x0 += [xf*i/N]
for i in range(N):
    x0 += [0.0]
for i in range(N):
    x0 += [0.0]
for i in range(N*9):
    x0 += [1e-12]
for i in range(3*N):
    x0 += [0.0]

# Solve the optimization problem
sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
print(sol.keys())

x_opt = sol['x'].full().flatten()
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

u = np.cos(phi_vals)*np.cos(theta_vals)
v = np.sin(phi_vals)*np.cos(theta_vals)
w = np.sin(theta_vals)
norm = np.sqrt(u**2 + v**2 + w**2)
u, v, w = u/norm, v/norm, w/norm  # Normalize the direction vectors

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(projection='3d')
ax.plot(x_vals, y_vals, z_vals, label='Trajectory')
ax.plot([0, xf], [0, 0], [0, 0], color='black', linestyle='--', label='Target Path')
ax.scatter(0, 0, 0, color='green', label='Start')
ax.scatter(xf, 0, 0, color='red', label='End')
# ax.quiver(x_vals, y_vals, z_vals, vx_vals, vy_vals, vz_vals, length=0.5, normalize=True, color='orange', label='Velocity')
ax.quiver(x_vals, y_vals, z_vals, u, v, w, length=0.5, normalize=True, color='blue', label='Orientation')
ax.legend()
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(9, 1, 1)
ax.plot(x_vals, label='x position'),ax.legend()
ax = fig.add_subplot(9, 1, 2)
ax.plot(y_vals, label='y position'),ax.legend()
ax = fig.add_subplot(9, 1, 3)
ax.plot(z_vals, label='z position'),ax.legend()
ax = fig.add_subplot(9, 1, 4)
ax.plot(vx_vals, label='vx'),ax.legend()
ax = fig.add_subplot(9, 1, 5)
ax.plot(vy_vals, label='vy'),ax.legend()
ax = fig.add_subplot(9, 1, 6)   
ax.plot(vz_vals, label='vz'),ax.legend()
ax = fig.add_subplot(9, 1, 7)
ax.plot(umag_vals, label='umag'),ax.legend()
ax = fig.add_subplot(9, 1, 8)
ax.plot(ul_vals, label='ul'),ax.legend()
ax = fig.add_subplot(9, 1, 9)
ax.plot(ur_vals, label='ur'),ax.legend()


plt.show()


# if __name__ == "__main__":
#     # Example usage of the dynamics function
#     s = get_state(0)  # Get the state at the first time step
#     u = get_control(0)  # Get the control at the first time step
#     dynamics.rhs_symbolic(s, u)  # Print the dynamics output for the first state and control