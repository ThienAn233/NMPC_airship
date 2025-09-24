import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import time
import pybullet_data
import numpy as np
import utils
import yaml




class AirshipEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}



    def __init__(self,fps=24,eplen=500,render_mode=None,config_file='NMPC_airship//RL//config.yaml',seed = 0):
        super(AirshipEnv).__init__()
        self.fps            = fps
        self.eplen          = eplen
        self.render_mode    = render_mode
        self.config_file    = config_file
        self.seed           = seed


        ### CONFIG LOADING ###
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        ### CONFIG LOADING ###


        ### RENDER SETTINGS ####
        if render_mode:
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
            self.linex = p.addUserDebugLine((0,0,0)             ,(1,0,0)                ,lineColorRGB=(1,0,0),physicsClientId=self.physicsClient)
            self.liney = p.addUserDebugLine((0,0,0)             ,(0,1,0)                ,lineColorRGB=(0,1,0),physicsClientId=self.physicsClient)
            self.linez = p.addUserDebugLine((0,0,0)             ,(0,0,1)                ,lineColorRGB=(0,0,1),physicsClientId=self.physicsClient)
            self.line1 = p.addUserDebugLine(self.config['rp_r'] ,self.config['rp_r']    ,lineColorRGB=(1,0,0),physicsClientId=self.physicsClient)
            self.line2 = p.addUserDebugLine(self.config['rp_l'] ,self.config['rp_l']    ,lineColorRGB=(1,0,0),physicsClientId=self.physicsClient)
            self.linet = p.addUserDebugLine(( 0,    0,    0)    ,( 0,    0,    0)       ,lineColorRGB=(1,0,0),physicsClientId=self.physicsClient)
            self.lined = p.addUserDebugLine(( 0,    0,    0)    ,( 0,    0,    0)       ,lineColorRGB=(0,1,0),physicsClientId=self.physicsClient)
            self.linew = p.addUserDebugLine(( 0,    0,    0)    ,( 0,    0,    0)       ,lineColorRGB=(0,0,1),physicsClientId=self.physicsClient)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        ### RENDER SETTINGS ####


        ### DYNAMIC CONST ###
        self.time_steps_in_current_episode  = 1
        self.action                         = np.array([0, 0, 0, 0])
        self.previous_action                = np.array([0, 0, 0, 0])
        #########################################################################################################################################################
        self.m_air      = self.config['rho'] * self.config['volume']  # displaced air mass
        self.mass       = self.config['mass'] + self.m_air
        self.I_added    = self.m_air * np.array([0.0, self.config['k3'], self.config['k3']])  # eq.[I'_0] eq 42
        self.I          = np.array([self.config['Ixx'], self.config['Iyy'], self.config['Izz']]) + self.I_added
        #########################################################################################################################################################
        self.good_dir = 0 
        self.good_fac = 0
        self.good_alt = 0
        self.reaching = 0
        self.living   = 0
        ### DYNAMIC CONST ###   


        ### SETUP ENV ###
        np.random.seed(self.seed)
        p.setGravity(0,0,0,physicsClientId=self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=self.physicsClient)
        pos = self.config['startPos']
        ori = p.getQuaternionFromEuler(self.config['startOri'],physicsClientId=self.physicsClient)
        # self.planeId    = p.loadURDF(self.config['plane_PATH'],physicsClientId=self.physicsClient,globalScaling=100)
        self.ShapeId    = p.createVisualShape(shapeType=p.GEOM_MESH,
                                              fileName=self.config['airshipobj'],
                                              meshScale=[3.5,3.5,3.5],
                                              rgbaColor=[.5,.5, .5, .7],
                                              visualFramePosition=-np.array(self.config['rg']))
        self.AirShip    = p.createMultiBody(baseMass=1,
                                            baseVisualShapeIndex=self.ShapeId,
                                            basePosition=pos,
                                            baseOrientation=ori,
                                            baseInertialFramePosition=-np.array(self.config['rg']),
                                            physicsClientId=self.physicsClient)
        p.changeDynamics(self.AirShip, -1, mass=self.mass,localInertiaDiagonal=self.I, physicsClientId=self.physicsClient)
        self.targetId   = p.loadURDF(self.config['target_PATH'],physicsClientId=self.physicsClient)
        self.reset()
        ### SETUP ENV ###
        
        
        ### SETUP SPACES ###
        self.action_space       = spaces.Box(low=-1,high=1,shape=(self.config['no_thrusters']+1,),seed=self.seed)
        self.observation_space  = spaces.Box(low=-3.4e+38,high=3.4e+38,shape=(len(self.get_obs()),),seed=self.seed)
        ### SETUP SPACES ###
        
        print(f'Environment created with seed: {self.seed}')
        return None
    
    
    
    def get_obs(self):
        self.pos, self.ori = p.getBasePositionAndOrientation(self.AirShip,physicsClientId=self.physicsClient)
        self.lin, self.ang = p.getBaseVelocity(self.AirShip,physicsClientId=self.physicsClient)
        self.lin, self.ang = np.array(list(self.lin)+[0]), np.array(list(self.ang)+[0]) 
        self.lin, self.ang = utils.active_rotation(np.array(self.ori),self.lin)[:3], utils.active_rotation(np.array(self.ori),self.ang)[:3]
        self.wind          = np.array(self.Windmodel()+[0])
        self.wind          = utils.active_rotation(np.array(self.ori),self.wind)[:3]  
        self.winlin        = self.lin - self.wind   
        self.dir           = utils.active_rotation(np.array(self.ori),np.array(list(self.targetPos-self.pos)+[0]))[:3]
        self.dir           = self.dir/np.linalg.norm(self.dir)
        self.rot           = utils.passive_rotation(np.array(self.ori),[0,0,1,0])[:3]
        self.fac           = utils.passive_rotation(np.array(self.ori),[1,0,0,0])[:3]
        self.sid           = utils.passive_rotation(np.array(self.ori),[0,1,0,0])[:3]
        obs                = np.array([*self.dir,*self.rot,*self.lin,*self.ang,*self.action,*self.previous_action])
        return obs.astype('float32')
    
    
    
    def get_info(self):
        return {'r1':self.good_dir,'r2':self.good_fac,'r3':self.reaching,'r4':self.living}
    
    
    
    def get_reward(self):
        ### Reward for good rotational dir  ###
        self.good_dir = (self.rot/np.linalg.norm(self.rot))[-1]-0.9
        ### Reward for facing good dir      ###
        self.good_fac = np.dot(self.fac,self.targetPos-self.pos)/(np.linalg.norm(self.fac)*np.linalg.norm(self.targetPos-self.pos)) - 0.8
        ### Reward for reaching the target  ###
        self.reaching = self.calculate_reward() #-np.linalg.norm(self.pos-self.targetPos) #
        ### Reward for living               ###
        self.living   = 1
        flip = (self.rot/np.linalg.norm(self.rot))[-1] 
        fell = self.pos[-1] 
        goal = np.linalg.norm(self.pos-self.targetPos) 
        if goal < 10:
            self.living*=1000000
        if (flip < -1 ) or (fell < 8):
            self.living *= -1000000
        return self.config['reward_coeff'][0]*self.good_dir + self.config['reward_coeff'][1]*self.good_fac  +self.config['reward_coeff'][2]*self.reaching + self.config['reward_coeff'][3]*self.living
    
    

    def calculate_reward(self):
        l = np.linalg.norm(np.cross(self.pos-np.array(self.config['startPos']),self.targetPos-np.array(self.config['startPos'])))/np.linalg.norm(self.targetPos-np.array(self.config['startPos']))
        da = np.linalg.norm(self.pos-self.targetPos)
        const = np.linalg.norm(self.targetPos-np.array(self.config['startPos']))
        return -20*l - 20*(np.abs(da) - const)


    
    def get_term_n_trunc(self):
        termination = False
        truncation  = False
        flip = (self.rot/np.linalg.norm(self.rot))[-1] 
        fell = self.pos[-1] 
        goal = np.linalg.norm(self.pos-self.targetPos) 
        tout = self.time_steps_in_current_episode > self.eplen
        if (flip < -1 ) or (fell < 8) or goal < 10:
            termination =True
        if tout:
            truncation  = True 
        return termination, truncation
    


    def reset(self,seed=None,options=None):
        if seed:
            self.seed = seed
        ori = self.config['startOri']
        ori+= np.random.uniform(*self.config['orirand'],size=(3,))
        pos = self.config['startPos']
        pos+= np.random.uniform(*self.config['posrand'],size=(3,))
        lin = np.random.uniform(*self.config['linvelrand'],size=(3,))
        ang = np.random.uniform(*self.config['angvelrand'],size=(3,))
        phi = np.random.uniform(0,2*np.pi)
        z   = np.random.uniform(*self.config['target_range'][0])
        r   = self.config['target_range'][1]
        self.targetPos = np.array([r*np.sin(phi), r*np.cos(phi),z])
        self.windphase = np.random.uniform(0,2*np.pi)
        p.resetBasePositionAndOrientation(self.AirShip,pos,p.getQuaternionFromEuler(ori),physicsClientId=self.physicsClient)
        p.resetBasePositionAndOrientation(self.targetId,self.targetPos,[0,0,0,1],physicsClientId=self.physicsClient)
        p.resetBaseVelocity(self.AirShip,lin,ang,physicsClientId=self.physicsClient)
        self.time_steps_in_current_episode = 1
        return  self.get_obs(), self.get_info()
    
    
    
    def step(self,action):
        self.action = np.array(action)
        k = self.config['action_filter']
        thrust = k*self.action+(1-k)*self.previous_action
        wind = self.Windmodel()
        for _ in range(10):
            ##### PHYSICAL MODEL #####
            ### GRAVITY ###
            p.applyExternalForce(self.AirShip,-1,(0,0,self.config['mass']*self.config['g']),self.pos,p.WORLD_FRAME,physicsClientId=self.physicsClient)
            ### GRAVITY ###
            
            
            ### BUOYANCY ###
            Rz = utils.quaternion_matrix(self.ori)
            fb_ERF = -np.array([[0],[0],[self.config['rho'] * self.config['volume'] * self.config['g']]])
            fb_BRF = (Rz.T@fb_ERF).T.flatten()
            p.applyExternalForce(self.AirShip,-1,fb_BRF,self.config['rg'],p.LINK_FRAME,physicsClientId=self.physicsClient)
            ### BUOYANCY ###

            ### PROPULSION ###
            T_mag = thrust[0]
            mu = thrust[1]
            nu = thrust[2]
            # 计算右侧推力向量
            thrust_vector_r = np.vstack(
                [T_mag * np.cos(mu) * np.cos(nu),
                T_mag * np.sin(mu),
                T_mag * np.cos(mu) * np.sin(nu)]).T.flatten()
            # 计算左侧推力向量
            thrust_vector_l = np.vstack(
                [T_mag * np.cos(mu) * np.cos(nu),
                T_mag * np.sin(mu),
                T_mag * np.cos(mu) * np.sin(nu)]).T.flatten()
            p.applyExternalForce(self.AirShip,-1,thrust_vector_r.flatten(),self.config['rp_r'],p.LINK_FRAME,physicsClientId=self.physicsClient)
            p.applyExternalForce(self.AirShip,-1,thrust_vector_l.flatten(),self.config['rp_l'],p.LINK_FRAME,physicsClientId=self.physicsClient)
            ### PROPULSION ###

            ### AERO DYNAMIC FORCES ###
            V_wind_ERF = np.array([0.0, 0.0, 0.0])  # Wind speed in Earth coordinates
            V_wind_BRF = Rz.T @ V_wind_ERF  # Convert wind speed to aircraft coordinate system
            # Dynamic pressure
            q_dyn = 0.5 * self.config['rho'] * np.linalg.norm(self.lin - V_wind_BRF) ** 2
            # Calculating relative speed
            u_rel, v_rel_body, w_rel = (self.lin - V_wind_BRF)[0], (self.lin - V_wind_BRF)[1], (self.lin - V_wind_BRF)[2]
            # Angle of attack and sideslip angle
            alpha = np.atan2(w_rel, u_rel)  # calculate relative wind speed magnitude (if not provided)
            V_rel_mag = np.sqrt(u_rel ** 2 + v_rel_body ** 2 + w_rel ** 2)
            beta = np.asin(v_rel_body / (V_rel_mag + 1e-6))  # calculate side slip angle
            # Calculate aerodynamic forces and moments using the extracted functions
            X_a = -q_dyn * (self.config['C_x1'] * np.cos(alpha) ** 2 * np.cos(beta) ** 2 + self.config['C_x2'] * np.sin(2 * alpha) * np.sin(alpha / 2))
            Y_a = -q_dyn * (self.config['C_y1'] * np.cos(beta / 2) * np.sin(2 * beta) + self.config['C_y2'] * np.sin(2 * beta) + self.config['C_y3'] * np.sin(beta) * np.sin(np.fabs(beta)))
            Z_a = -q_dyn * (self.config['C_z1'] * np.cos(alpha / 2) * np.sin(2 * alpha) + self.config['C_z2'] * np.sin(2 * alpha) + self.config['C_z3'] * np.sin(alpha) * np.sin(np.fabs(alpha)))
            fa_BRF = np.vstack([X_a, Y_a, Z_a]).T.flatten()
            p.applyExternalForce(self.AirShip, -1, fa_BRF, self.config['rg'], p.LINK_FRAME, physicsClientId=self.physicsClient)
            ### AERO DYNAMIC FORCES ###
            ##### PHYSICAL MODEL #####
            p.stepSimulation(physicsClientId=self.physicsClient)

        
        if self.render_mode:
            self.sleeper(self.config['f']*10)
            p.resetDebugVisualizerCamera(60,0,0,self.pos,physicsClientId=self.physicsClient)
            p.addUserDebugLine(self.config['rp_r'],self.config['rp_r']+thrust_vector_r*10,lineColorRGB=(1,0,0),parentObjectUniqueId=self.AirShip,parentLinkIndex=-1,replaceItemUniqueId=self.line1,physicsClientId=self.physicsClient)
            p.addUserDebugLine(self.config['rp_l'],self.config['rp_l']+thrust_vector_l*10,lineColorRGB=(1,0,0),parentObjectUniqueId=self.AirShip,parentLinkIndex=-1,replaceItemUniqueId=self.line2,physicsClientId=self.physicsClient)
            # p.addUserDebugLine((0,0,0)                    ,aero                                                   ,lineColorRGB=(1,0,0),parentObjectUniqueId=self.AirShip,parentLinkIndex=-1,replaceItemUniqueId=self.linet,physicsClientId=self.physicsClient)
            # p.addUserDebugLine((0,0,0)                    ,self.wind*5                                            ,lineColorRGB=(0,0,1),parentObjectUniqueId=self.AirShip,parentLinkIndex=-1,replaceItemUniqueId=self.linew,physicsClientId=self.physicsClient)
            # p.addUserDebugLine((0,0,0)                    ,self.lin*1                                             ,lineColorRGB=(0,1,0),parentObjectUniqueId=self.AirShip,parentLinkIndex=-1,replaceItemUniqueId=self.lined,physicsClientId=self.physicsClient)
            # p.addUserDebugLine(self.pos+np.array([-10,-30,7])*0, self.pos + self.fac +np.array([-10,-30,7])*0         ,lineColorRGB=(1,0,0),replaceItemUniqueId=self.linex,physicsClientId=self.physicsClient)
            # p.addUserDebugLine(self.pos+np.array([-10,-30,7])*0, self.pos + self.rot +np.array([-10,-30,7])*0         ,lineColorRGB=(0,1,0),replaceItemUniqueId=self.liney,physicsClientId=self.physicsClient)
            # p.addUserDebugLine(self.pos+np.array([-10,-30,7])*0, self.pos + self.sid +np.array([-10,-30,7])*0         ,lineColorRGB=(0,0,1),replaceItemUniqueId=self.linez,physicsClientId=self.physicsClient)
        p.resetBasePositionAndOrientation(self.targetId,self.targetPos, [0,0,0,1],physicsClientId=self.physicsClient)
        self.previous_action = self.action
        self.time_steps_in_current_episode += 1
        return self.get_obs(), self.get_reward(), *self.get_term_n_trunc(), self.get_info()




    def sleeper(self,sleep):
        to = time.time()
        while time.time()-to<sleep:
            pass
    


    def Windmodel(self):
        base_wind = 2.0 * (1 + self.pos[-1]/1000)  # m/s
        # Add periodic wind speed changes in three directions
        # t is time, using sine and cosine functions to simulate the periodic change of wind speed
        wind_x = base_wind * np.sin(0.001*self.time_steps_in_current_episode*self.config['f']+self.windphase)    # Wind speed in x direction, cycle is about 6280 seconds (1.7 hours)
        wind_y = base_wind * np.cos(0.001*self.time_steps_in_current_episode*self.config['f']+self.windphase)    # Wind speed in the y direction, phase difference from x direction is 90 degrees
        wind_z = 0.5 * base_wind * np.sin(0.0005*self.time_steps_in_current_episode*self.config['f']+self.windphase)   # The wind speed in the direction of z, with smaller amplitude and longer periods
        return [wind_x,wind_y,wind_z]



if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import SAC
    env = AirshipEnv(render_mode = 'human')
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    print('checked, no error!')
    # model = SAC.load('AirshipControl/airship_rl/training/model/SAC_airship_2025-04-02-11-12-18/model.zip',device='cpu',print_system_info=True)
    obs, info = env.reset()
    # p.resetDebugVisualizerCamera(60,0,0,env.pos,physicsClientId=env.physicsClient)
    t = 0
    while True:
        # time.sleep(0.2)
        # action, _ = model.predict(obs, deterministic=False)
        action = np.random.uniform(-1,1,(env.action_space.shape))
        # action = np.zeros(shape=env.action_space.shape)
        obs,reward,terminated,truncated,info = env.step(action)
        print(reward,info)
        if terminated or truncated:
            env.reset()