import yaml

# Data to be written to the YAML file
data = {
    'plane_PATH'    : "plane.urdf",
    'airship_PATH'  : "AirshipControl//airship_rl//airship_model//urdf//airship.urdf",
    'target_PATH'   : "AirshipControl//airship_rl//airship_model//urdf//target.urdf",
    'g'             : -9.81,
    'f'             : 1./240.,
    'startPos'      : [0,0,100],
    'startOri'      : [0,0,0],
    'mass'          : 474.181,
    'no_thrusters'  : 3,
    'thrust'        : 10000,
    'rho'           : 0.089,
    'length'        : 16, 
    'diameter'      : 8,
    'Cd'            : [0.02, 0.02, 0.03],
    'Cl'            : [0.2, 0.2, 0.2],
    'ThrusterPos1'  : [-6,    0,  3.5],
    'ThrusterPos2'  : [-6,-3.03,-1.75],
    'ThrusterPos3'  : [-6, 3.03,-1.75],
    'posrand'       : [-1,1],
    'orirand'       : [-0.2, 0.2],
    'linvelrand'    : [-1,1],
    'angvelrand'    : [-0.2,0.2],
    'target_range'  : [[80,120],100], 
    'reward_coeff'  : [1000,1000,1,1],
    'action_filter' : 0.5,
    'buoyancy_gain' : 5000,
    'thrust_gain'   : 2000,
    'center_of_mass': [-0.576,0,-0.366],
}


with open('AirshipControl//airship_rl//config.yaml', 'w') as file:
    yaml.dump(data, file)

print("Data has been written to 'data.yaml'")