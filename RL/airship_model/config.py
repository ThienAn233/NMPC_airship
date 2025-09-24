import yaml

# Data to be written to the YAML file
data = {
    'plane_PATH'    : "plane.urdf",
    'airship_PATH'  : "NMPC_airship//RL//airship_model//urdf//airship.urdf",
    'airshipobj'    : "NMPC_airship//RL//airship_model//meshes/airship.obj",
    'target_PATH'   : "NMPC_airship//RL//airship_model//urdf//target.urdf",
    'g'             : -9.74,
    'f'             : 1./240.,
    'startPos'      : [0,0,2000],
    'startOri'      : [0,0,0],
    'mass'          : 2934,
    'volume'        : 35705,
    'Ixx'           : 393187,
    'Iyy'           : 1224880,   
    'Izz'           : 939666,
    'airship_a1'    : 88.70/2,  
    'airship_a2'    : 88.70/2,
    'airship_b'     : 27.7/2,
    'rg'            : [0, 0, 2.66],    
    'rb'            : [0, 0, 0],
    'rp_r'          : [0, 27.7/2, -3],
    'rp_l'          : [0, -27.7/2, -3],
    'rho'           : 0.0822,
    'C_l1'          : 2.4e4 / 28.8,
    'C_m1'          : 7.7e4 / 28.8,
    'C_m2'          : 7.7e4 / 28.8,
    'C_m3'          : 7.7e4 / 28.8,
    'C_m4'          : 7.7e4 / 28.8,
    'C_n1'          : 7.7e4 / 28.8,
    'C_n2'          : 7.7e4 / 28.8,
    'C_n3'          : 7.7e4 / 28.8,
    'C_n4'          : 7.7e4 / 28.8,
    'C_x1'          : 657.0 / 28.8,
    'C_x2'          : 657.0 / 28.8,
    'C_y1'          : 657.0 / 28.8,
    'C_y2'          : 657.0 / 28.8,
    'C_y3'          : 657.0 / 28.8,
    'C_y4'          : 657.0 / 28.8,
    'C_z1'          : 657.0 / 28.8,
    'C_z2'          : 657.0 / 28.8,
    'C_z3'          : 657.0 / 28.8,
    'C_z4'          : 657.0 / 28.8,
    'k3'            : 0.53,  

    'no_thrusters'  : 3,
    'posrand'       : [-1,1],
    'orirand'       : [-0.2, 0.2],
    'linvelrand'    : [-1,1],
    'angvelrand'    : [-0.2,0.2],
    'target_range'  : [[2010,1990],200], 
    'reward_coeff'  : [1000,1000,1,1],
    'action_filter' : 0.5,
    'buoyancy_gain' : 5000,
    'thrust_gain'   : 2000,
}


with open('NMPC_airship//RL//config.yaml', 'w') as file:
    yaml.dump(data, file)

print("Data has been written to 'data.yaml'")