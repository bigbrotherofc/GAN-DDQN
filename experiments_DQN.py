'''
Author: bigbrotherofc 2903204960@qq.com
Date: 2024-01-13 12:04:59
LastEditors: bigbrotherofc 2903204960@qq.com
LastEditTime: 2024-01-19 16:15:00
FilePath: \GAN-DDQN\DQN.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE 
'''
from ctypes import util
import re
import sys
import os
import cellular_env
from numpy.random import default_rng
from cellular_env import cellularEnv
#from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DQN
import torch
import numpy as np
import gym 
import itertools
import matplotlib.pyplot as plt


def action_space(total, num):
    tmp = list(itertools.product(range(total + 1), repeat=num))
    result = []
    for value in tmp:
        if sum(value) == total:
            result.append(list(value))
    result = np.array(result)
    [i, j] = np.where(result == 0)
    result = np.delete(result, i, axis=0)
    print(result.shape)#(36,3)只有36个动作，这个动作空间太小了吧
    return result

if __name__ == '__main__': 

    device =torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    total_timesteps = 10000

    # parameters of celluar environment
    ser_cat_vec = ['volte', 'embb_general', 'urllc']
    band_whole_no = 10 * 10**6 # 10 MHz
    band_per = 1 * 10**6  # 1 MHz 离谱了吧，这个粒度太大了
    qoe_weight = [1, 1, 1] # qoe weights of each service category
    se_weight = 0.01 # weight of spectral efficiency
    dl_mimo = 64 # downlink mimo
    learning_window = 2000 # learning window
    path ='./result/DQN/'
    env = cellularEnv(ser_cat=ser_cat_vec, learning_windows=learning_window, dl_mimo=dl_mimo,path=path)
    action_space = action_space(10, 3) * band_per
    num_actions = len(action_space) #,这里就在后面定了动作空间的大小，以及DQN的输入大小
    env = gym.make('GymCelluarEnv-v1', cellularEnv=env, ser_cat_action_space=action_space, qoe_weight=qoe_weight, se_weight=se_weight,total_timesteps=total_timesteps)
    #env = make_vec_env(lambda: env, n_envs=1)
    
    agent=DQN('MlpPolicy',env,verbose=True) #stablebaselins3 中的智能体和gymnasuim 中的环境是是适应找到gym的环境然后是我们自己包装的。
    agent.learn(total_timesteps=total_timesteps)
    
    plt.figure(1)
    plt.clf()
    rewards = np.array(env.rewards)
    utilitys = np.array(env.utilitys)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(utilitys)
    plt.show()