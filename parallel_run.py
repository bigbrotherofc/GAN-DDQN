from experiments_DQN import *
import math
import os
from numpy.random import default_rng
from itertools import product
import concurrent.futures as cf

from stable_baselines3 import PPO1, PPO2, TRPO, SAC, A2C, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env


PROCESSES = 4 # 30 if enough threads 
device =torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
total_timesteps = 1000
epsilon_start    = 1.0
epsilon_final    = 0.01
epsilon_decay    = 3000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# parameters of celluar environment
ser_cat_vec = ['volte', 'embb_general', 'urllc']
band_whole_no = 10 * 10**6 # 10 MHz
band_per = 1 * 10**6  # 1 MHz 离谱了吧，这个粒度太大了
qoe_weight = [1, 1, 1] # qoe weights of each service category
se_weight = 0.01 # weight of spectral efficiency
dl_mimo = 64 # downlink mimo
learning_window = 2000 # learning window
path ='./result/'
action_space = action_space(10, 3) * band_per
num_actions = len(action_space) #,这里就在后面定了动作空间的大小，以及DQN的输入大小
algorithms = {
    'SAC':SAC,
    'PPO1':PPO1, 
    'PPO2':PPO2, 
    'TRPO':TRPO,
    'A2C': A2C,
    'TD3': TD3,
    'DDPG': DDPG
}

deterministic = {
    'SAC':True,
    'PPO1':True, 
    'PPO2':False, 
    'TRPO':False,
    'A2C':False,
    'TD3':False,
    'DDPG':False
}
class RLEvaluator():
    def __init__(self, scenario, algo_name, algorithm):
        self.scenario = scenario
        self.algo_name = algo_name
        self.algorithm = algorithm
        self.path = './results/{}/'.format(algo_name)
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print ("Creation of the directory %s failed" % self.path)
            else:
                print ("Successfully created the directory %s " % self.path)
        self.model_path = './saved_agents/{}/'.format(algo_name)
        if not os.path.isdir(self.model_path):
            try:
                os.makedirs(self.model_path)
            except OSError:
                print ("Creation of the directory %s failed" % self.model_path)
            else:
                print ("Successfully created the directory %s " % self.model_path)
    def evaluate(self, seed):
        env = cellularEnv(ser_cat=ser_cat_vec, learning_windows=learning_window, dl_mimo=dl_mimo,path=path)
        env = gym.make('GymCelluarEnv-v1', cellularEnv=env, ser_cat_action_space=action_space, qoe_weight=qoe_weight, se_weight=se_weight,total_timesteps=total_timesteps)
        env.seed(seed)
        model = self.algorithm('MlpPolicy', env, verbose=1, device=device)
        model.learn(total_timesteps=total_timesteps)
        model.save(self.model_path + '{}_{}_{}'.format(self.scenario, self.algo_name, seed))
        env.close()
        return seed, model   
        