import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import do_mpc
import time as t
from airship_dynamic import AirshipCasADiSymbolic

n_horizon   = 10
dt          = 0.1
n_robust    = 1
v_max       = 1
T_max       = 100000
L           = ca.diag([1e4,1e8,1e8])
Q           = ca.diag([1e-4/T_max,1e-4,1e-4])
target      = ca.vertcat(9, 0, 0)
target_ori  = ca.vertcat([0, 0, np.pi])

model_type = 'continuous'
model = do_mpc.model.Model(model_type)

# x           = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
# y           = model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
# z           = model.set_variable(var_type='_x', var_name='z', shape=(1, 1))
# phi         = model.set_variable(var_type='_x', var_name='phi', shape=(1, 1))
# theta       = model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
# psi         = model.set_variable(var_type='_x', var_name='psi', shape=(1, 1))
# x_dot       = model.set_variable(var_type='_x', var_name='x_dot', shape=(1, 1))
# y_dot       = model.set_variable(var_type='_x', var_name='y_dot', shape=(1, 1))
# z_dot       = model.set_variable(var_type='_x', var_name='z_dot', shape=(1, 1))
# phi_dot     = model.set_variable(var_type='_x', var_name='phi_dot', shape=(1, 1))
# theta_dot   = model.set_variable(var_type='_x', var_name='theta_dot', shape=(1, 1))
# psi_dot     = model.set_variable(var_type='_x', var_name='psi_dot', shape=(1, 1))

# u_mag       = model.set_variable(var_type='_u', var_name='u_mag', shape=(1, 1))
# u_l         = model.set_variable(var_type='_u', var_name='u_l', shape=(1, 1))
# u_r         = model.set_variable(var_type='_u', var_name='u_r', shape=(1, 1))

# s = model.set_variable(var_type='_x', var_name='s', shape=(12, 1)) 
u = model.set_variable(var_type='_u', var_name='u', shape=(3, 1))  

pos = model.set_variable(var_type='_x', var_name='pos', shape=(3, 1))  # [x, y, z]
ori = model.set_variable(var_type='_x', var_name='ori', shape=(3, 1))  # [phi, theta, psi]
vel = model.set_variable(var_type='_x', var_name='vel', shape=(3, 1))  # [vx, vy, vz]
orv = model.set_variable(var_type='_x', var_name='orv', shape=(3, 1))  # [omega_phi, omega_theta, omega_psi]

s = ca.vertcat(pos, ori, vel, orv)  # State vector
dynamics = AirshipCasADiSymbolic()

# model.set_rhs('s',dynamics.rhs_symbolic(s, u))  
model.set_rhs('pos', dynamics.rhs_symbolic(s,u)[0:3])  # [x, y, z]
model.set_rhs('ori', dynamics.rhs_symbolic(s,u)[3:6])  # [phi, theta, psi]
model.set_rhs('vel', dynamics.rhs_symbolic(s,u)[6:9])  # [vx, vy, vz]
model.set_rhs('orv', dynamics.rhs_symbolic(s,u)[9:12])  # [omega_phi, omega_theta, omega_psi]
model.setup()

mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': n_horizon,
    't_step': dt,
    'n_robust': n_robust,
    'store_full_solution': True,
    
}
mpc.set_param(**setup_mpc)
mpc.settings.supress_ipopt_output() 

dis = pos - target
rot = ori - target_ori
mterm = dis.T@L@dis + rot.T@L@rot
lterm = dis.T@L@dis + rot.T@L@rot
rterm = u.T@Q@u
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(rterm=rterm)

# Lower bounds on states:
mpc.bounds['lower','_x','ori',0] = -2*np.pi
mpc.bounds['lower','_x','ori',1] = -2*np.pi
mpc.bounds['lower','_x','ori',2] = -2*np.pi
mpc.bounds['lower','_x','vel',0] = -v_max
mpc.bounds['lower','_x','vel',1] = -v_max
mpc.bounds['lower','_x','vel',2] = -v_max
# Upper bounds on states
mpc.bounds['upper','_x','ori',0] = 2*np.pi
mpc.bounds['upper','_x','ori',1] = 2*np.pi
mpc.bounds['upper','_x','ori',2] = 2*np.pi
mpc.bounds['upper','_x','vel',0] = v_max
mpc.bounds['upper','_x','vel',1] = v_max    
mpc.bounds['upper','_x','vel',2] = v_max
# Lower bounds on inputs:
mpc.bounds['lower','_u','u',0] = 0
mpc.bounds['lower','_u','u',1] = -np.pi
mpc.bounds['lower','_u','u',2] = -np.pi
# Lower bounds on inputs:
mpc.bounds['upper','_u','u',0] = T_max
mpc.bounds['upper','_u','u',1] = np.pi
mpc.bounds['upper','_u','u',2] = np.pi

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)

simulator.set_param(t_step = dt)
simulator.setup()

x0 = ca.vertcat([0., 0., 0., 0., 0., np.pi, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

results = []
for i in range(50):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    results += [x0]
    print(f"[{mpc.solver_stats['return_status']}][dt: {round(i*dt,2)}]: x position: {x0[0]}, y position: {x0[1]}, z position: {x0[2]}, phi: {x0[3]}, theta: {x0[4]}, psi: {x0[5]}")
    t.sleep(0.5)


# print(results)
x = [res[0] for res in results]
y = [res[1] for res in results]
z = [res[2] for res in results]
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Airship Path')
ax.scatter(x, y, z, c='r', marker='o', label='Airship Position')
ax.scatter([target[0]], [target[1]], [target[2]], c='g', marker='x', label='Target Position')
ax.scatter([0], [0], [0], c='b', marker='^', label='Start Position')
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Airship Trajectory')
ax.legend()
plt.show()