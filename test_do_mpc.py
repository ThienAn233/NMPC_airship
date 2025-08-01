import numpy as np
import casadi as ca
import do_mpc
from airship_dynamic import AirshipCasADiSymbolic

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

s = model.set_variable(var_type='_x', var_name='s', shape=(12, 1)) 
u = model.set_variable(var_type='_u', var_name='u', shape=(3, 1))  # [T_mag, mu, nu]
dynamics = AirshipCasADiSymbolic()
model.set_rhs('s',dynamics.rhs_symbolic(s, u))
model.setup()
