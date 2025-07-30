import casadi as ca
import numpy as np
from airship_dynamic import AirshipCasADiSymbolic

x = ca.SX.sym('x')
y = ca.SX.sym('y')
z = ca.SX.sym('z')
phi = ca.SX.sym('phi')
theta = ca.SX.sym('theta')
psi = ca.SX.sym('psi')
vx = ca.SX.sym('vx')
vy = ca.SX.sym('vy')    
vz = ca.SX.sym('vz')
omega_phi = ca.SX.sym('omega_phi')
omega_theta = ca.SX.sym('omega_theta')  
omega_psi = ca.SX.sym('omega_psi')
umag = ca.SX.sym('umag')
ul = ca.SX.sym('ul')
ur = ca.SX.sym('ur')

s = ca.vertcat(x, y, z, phi, theta, psi, vx, vy, vz, omega_phi, omega_theta, omega_psi)
u = ca.vertcat(umag, ul, ur)

dynamics = AirshipCasADiSymbolic()

s0 = ca.vertcat(
    0, 0, 0,  # Initial position (x, y, z)
    0, 0, np.pi,  # Initial orientation (phi, theta, psi)
    0, 0, 0,  # Initial velocities (vx, vy, vz)
    0, 0, 0,  # Initial angular velocities (omega_phi, omega_theta, omega_psi)
)
u0 = ca.vertcat(0, 0, 0)  # Initial control inputs (umag, ul, ur)

dyna = ca.Function('dynamics', [s, u], [dynamics.rhs_symbolic(s, u)])

state_list = [s0]

for i in range(30):
    s_next = dyna(s0, u0)
    s0 = s_next  # Update state for next iteration
    state_list.append(s0)

# print(state_list)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
x_vals = [state[0] for state in state_list]
y_vals = [state[1] for state in state_list]
z_vals = [state[2] for state in state_list]
ax.plot(x_vals, y_vals, z_vals)
ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')
print(x_vals)
print(y_vals)
print(z_vals)
plt.show()