import os
import pybullet as p
import numpy as np
import gymnasium as gym
import time as t
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor



algo = 'SAC_airship'
start_time  = t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())
parent_dir  = 'AirshipControl/airship_rl/training/model/'+algo+'_'+start_time
os.mkdir(parent_dir)
tensor_path = 'AirshipControl/airship_rl/training/log'
log_name    = algo+'_'+start_time
model_dir   = '/model' 


gym.register(
    id="airship",
    entry_point="airship_env:AirshipEnv",
    max_episode_steps=500)
def create_env():
    return DummyVecEnv([lambda: Monitor(gym.make("airship",seed=1)),
                        lambda: Monitor(gym.make("airship",seed=2)),
                        lambda: Monitor(gym.make("airship",seed=3)),
                        lambda: Monitor(gym.make("airship",seed=4)),])
env = create_env()


batch_size = 4096
model = SAC(policy="MlpPolicy",batch_size=batch_size,learning_rate= 1e-5,env=env,verbose=1,tensorboard_log=tensor_path)
# PATH = 'AirshipControl/airship_rl/training/model/SAC_airship_2025-04-01-04-35-30/model.zip'
# model.set_parameters(PATH)
# model = PPO("MlpPolicy", env,learning_rate=3e-5, verbose=1,batch_size=4096,tensorboard_log=tensor_path)
model.learn(10000000,tb_log_name=log_name,progress_bar=True)
model.save(parent_dir+model_dir)
print('saved as: ',parent_dir+model_dir)