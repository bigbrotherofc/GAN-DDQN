import numpy as np
import gym
from gym import spaces
import itertools

class GymCelluarEnv(gym.Env):
    
    #ser_cat_action_space shape(len(action_num),len(ser_cat))
    def __init__(self, cellularEnv=None,ser_cat_action_space=None, qoe_weight=[1, 4, 6], se_weight=0.01,total_timesteps=10000):
        self.env = cellularEnv# cellulrEnv初始化之后在传进来
        self.file_path = self.env.file_path
        self.qoe_weight = qoe_weight
        self.se_weight = se_weight
        self.ser_cat_action_space = ser_cat_action_space
        self.action_space = spaces.Discrete(len(ser_cat_action_space))
        self.ser_cat = self.env.ser_cat
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(len(self.ser_cat),), dtype=np.float32)
        self.total_timesteps = total_timesteps
        self.times = 0
        self.rewards=[]
        self.utilitys = []
    #辅助函数，对数据进行中心化
    def state_update(self,state, ser_cat):
        discrete_state = np.zeros(state.shape)
        if state.all() == 0:
            return discrete_state
        for ser_name in ser_cat:
            ser_index = ser_cat.index(ser_name)
            discrete_state[ser_index] = state[ser_index]
        discrete_state = (discrete_state-discrete_state.mean())/discrete_state.std()
        return discrete_state
    def calc_reward(self,qoe, se, threshold):
        utility = np.matmul(self.qoe_weight, qoe.reshape((3, 1))) + self.se_weight * se #这个奖励公式从何而来
        if threshold > 11: #threshold的意义在于哪里
            threshold = 11
        if utility < threshold:
            reward = 0
        else:
            reward = 1
        return utility, reward

    def reset(self):
        self.env.countReset()#用户的到达包数，用户数，时延数等等清零
        self.env.activity()#时间从零开始计算，流量激活
        return self.state_update(self.env.tx_pkt_no, self.env.ser_cat) #返回中心化数据
    
    def step(self, action):
        self.env.band_ser_cat = self.ser_cat_action_space[action] #每个服务给多少带宽
        for i in itertools.count():
            self.env.scheduling()
            self.env.provisioning()
            #一秒停一次啊,每一秒清空一次是不是有点离谱了，到达之类的都会清空的话 这会造成不连续啊
            if i == (self.env.learning_windows- 1):
                break
            else:
                self.env.bufferClear()
                self.env.activity()
        qoe, se = self.env.get_reward()
        threshold = 6 + 4 * self.times / (self.total_timesteps/1.5)
        utility, reward = self.calc_reward(qoe, se, threshold)
        observation = self.state_update(self.env.tx_pkt_no, self.env.ser_cat)#真正的更新在tx_pkt_no和env类的参数变化中
        info={'qoe':qoe,'se':se,'utility':utility,'reward':reward,'threshold':threshold,'action':action,'tx_pkt_no':self.env.tx_pkt_no}
        self.rewards.append(reward)
        self.utilitys.append(utility)
        print('GANDDQN=====episode: %d, utility: %.5f, reward: %.5f' % (self.times, utility, reward))
        print('action:',action,'tx_pkt_no:',self.env.tx_pkt_no,'qoe:',qoe,'se:',se)
        self.times += 1
        return observation, reward, False, info     
    def render(self):
        pass 
    def save_results(self):
        np.savez(self.file_path,rewards = self.rewards,
                                utilitys = self.utilitys)
    