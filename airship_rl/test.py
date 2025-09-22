import pybullet as p
import time
import pybullet_data
import numpy as np
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
g = -9.81
f = 1./240.
p.setGravity(0,0,g)
planeId = p.loadURDF("plane.urdf",globalScaling=10)
startPos = [0,0,30]
startOrientation = p.getQuaternionFromEuler([0,0,0])
AirShip = p.loadURDF("AirshipControl//airship_rl//airship_model//urdf//airship.urdf",startPos, startOrientation)
targetId= p.loadURDF('//MonoCopter//urdf_models//target.urdf',[0,0,1],[0,0,0,1],globalScaling=10)
target = [50,0,30]
p.resetBasePositionAndOrientation(AirShip, startPos,startOrientation)

mass        = 474.181
thrust      = 10000
rho         = 0.089
length      = 16 
diameter    = 8
vol         = (4/3)*np.pi*8*4*4
A           = np.array([
            length * diameter,  # x-direction side area
            length * diameter,  # y direction side area
            np.pi * (diameter/2)**2  # Positive area in the direction of z (circular cross-section)
        ])
Cd = np.array([0.02, 0.02, 0.03])

param0 = p.addUserDebugParameter('Buoyancy' ,mass*-g*0,mass*-g*10,mass*-g)
param1 = p.addUserDebugParameter('Thruster1',-thrust,thrust,0)
param2 = p.addUserDebugParameter('Thruster2',-thrust,thrust,0)
param3 = p.addUserDebugParameter('Thruster3',-thrust,thrust,0)

line1 = p.addUserDebugLine((-6,    0,  3.5),(-6,    0,  3.5),lineColorRGB=(1,0,0),parentObjectUniqueId=AirShip,parentLinkIndex=-1)
line2 = p.addUserDebugLine((-6,-3.03,-1.75),(-6,-3.03,-1.75),lineColorRGB=(1,0,0),parentObjectUniqueId=AirShip,parentLinkIndex=-1)
line3 = p.addUserDebugLine((-6, 3.03,-1.75),(-6, 3.03,-1.75),lineColorRGB=(1,0,0),parentObjectUniqueId=AirShip,parentLinkIndex=-1)
lined = p.addUserDebugLine(( 0,    0,    0),( 0,    0,    0),lineColorRGB=(0,1,0),parentObjectUniqueId=AirShip,parentLinkIndex=-1)
linew = p.addUserDebugLine(( 0,    0,    0),( 0,    0,    0),lineColorRGB=(0,0,1),parentObjectUniqueId=AirShip,parentLinkIndex=-1)


t = 0
while True:
    pos, ori        = p.getBasePositionAndOrientation(AirShip)
    linvel, rovel   = p.getBaseVelocity(AirShip)


    
    Buoyancy  = p.readUserDebugParameter(param0)
    Thruster1 = p.readUserDebugParameter(param1)
    Thruster2 = p.readUserDebugParameter(param2)
    Thruster3 = p.readUserDebugParameter(param3)


    ###########################################################################
    ### BUOYANCY ###
    p.applyExternalForce(AirShip,-1,(0,0,Buoyancy),pos,p.LINK_FRAME)
    ### BUOYANCY ###

    ### PROPULSION ###
    p.applyExternalForce(AirShip,-1,(Thruster1,0,0),(-6,    0,  3.5),p.WORLD_FRAME)
    p.applyExternalForce(AirShip,-1,(Thruster2,0,0),(-6,-3.03,-1.75),p.LINK_FRAME)
    p.applyExternalForce(AirShip,-1,(Thruster3,0,0),(-6, 3.03,-1.75),p.LINK_FRAME)
    ### PROPULSION ###

    ### DRAG ###
    drag = -0.5 * rho * Cd * A * linvel * np.abs(linvel)
    p.applyExternalForce(AirShip,-1,drag           ,(-0,    0,    0),p.LINK_FRAME)
    ### DRAG ###

    ### DISTURBANCE ###
    # The wind speed will increase linearly with height: on the ground it is 2m/s, and every 1000 meters rise is increased by 2m/s
    base_wind = 2.0 * (1 + pos[-1]/1000)  # m/s
    # Add a randomness of 20%
    base_wind  += np.random.uniform(0.8*base_wind,1.2*base_wind)
    freq        = f*1/6280
    wind_x      = base_wind * np.sin(freq*t)    # Wind speed in x direction, cycle is about 6280 seconds (1.7 hours)
    wind_y      = base_wind * np.cos(freq*t)    # Wind speed in the y direction, phase difference from x direction is 90 degrees
    wind_z      = 0.5 * base_wind * np.sin(freq/2*t)   # The wind speed in the direction of z, with smaller amplitude and longer periods
    ### DISTURBANCE ###
    ###########################################################################



    


    p.addUserDebugLine((-6,    0,  4),(-6+Thruster1/800,    0,  4),lineColorRGB=(1,0,0),parentObjectUniqueId=AirShip,parentLinkIndex=-1,replaceItemUniqueId=line1)
    p.addUserDebugLine((-6,-3.46,-2),(-6+Thruster2/800,-3.46,-2),lineColorRGB=(1,0,0),parentObjectUniqueId=AirShip,parentLinkIndex=-1,replaceItemUniqueId=line2)
    p.addUserDebugLine((-6, 3.46,-2),(-6+Thruster3/800, 3.46,-2),lineColorRGB=(1,0,0),parentObjectUniqueId=AirShip,parentLinkIndex=-1,replaceItemUniqueId=line3)
    p.addUserDebugLine(( 0,    0,    0),                       drag/10,lineColorRGB=(0,1,0),parentObjectUniqueId=AirShip,parentLinkIndex=-1,replaceItemUniqueId=lined)

    p.resetBasePositionAndOrientation(targetId,target,[0,0,0,1])
    p.stepSimulation()
    # time.sleep(f)
    p.resetDebugVisualizerCamera(40,0,0,pos)
    # p.getCameraImage(1024,1024)
    print(pos)
    t += 1
    if t > 1e6:
        t=0
p.disconnect()