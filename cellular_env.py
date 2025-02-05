'''
For downlink simulations in one-single base station environment


'''
import os
import numpy as np
import time

 
class cellularEnv(object):
    def __init__(self,
        rng,
        BS_pos = np.array([0,0]),
        BS_radius = 40,
        #BS_tx_power = 0, #unit is dBW
        BS_tx_power = 16, #unit is dBW, 46dBm
        UE_max_no = 100, 
        Queue_max = 5,
        noise_PSD = -204, # -174 dbm/Hz
        carrier_freq = 2 * 10 ** 9, #2 GHz
        chan_mod = '36814',
        time_subframe = 0.5 * 10 ** (-3), # by LTE, 0.5 ms
        ser_cat = ['volte','embb_general','urllc'],
        band_whole = 10 * 10 ** 6, # 10MHz
        schedu_method = 'round_robin',
        ser_prob = np.array([6,6,1], dtype=np.float32),
        dl_mimo = 32,
        rx_gain = 20, #dB
        learning_windows = 60000,
        path = 'data/'
        ):
        self.rng = rng
        self.BS_tx_power = BS_tx_power
        self.BS_radius = BS_radius
        self.band_whole = band_whole
        self.chan_mod = chan_mod
        self.carrier_freq = carrier_freq
        self.time_subframe = round(time_subframe,4)
        self.noise_PSD = noise_PSD
        self.sys_clock = 0
        self.schedu_method = schedu_method 
        self.dl_mimo = dl_mimo
        self.UE_rx_gain = rx_gain
        self.UE_max_no = UE_max_no
        self.UE_buffer = np.zeros([Queue_max,UE_max_no])#(5,100)
        self.UE_buffer_backup = np.zeros([Queue_max,UE_max_no])
        self.UE_latency = np.zeros([Queue_max,UE_max_no])
        self.UE_readtime = np.zeros(UE_max_no)
        self.UE_band = np.zeros(UE_max_no)
        UE_pos = self.rng.uniform(-self.BS_radius, self.BS_radius, [self.UE_max_no,2])#(100,2)
        dis = np.sqrt(np.sum((BS_pos - UE_pos) **2 , axis = 1)) / 1000 # unit changes to km distance
        self.path_loss = 145.4 + 37.5 * np.log10(dis).reshape(-1,1)
        self.learning_windows = round(learning_windows*self.time_subframe,4) #2000*0.5ms = 1s 保留四位小数
        self.ser_cat = ser_cat
        self.file_path = path+'/historys.npz'
        if not os.path.isdir(path):
            try:
                os.makedirs(path)
            except OSError:
                print('create file path failed{}'.format(path))
            else:
                print('create file path success{}'.format(path))
            
        if len(self.ser_cat) > 1:
            self.band_ser_cat = np.zeros(len(ser_cat))
            if len(ser_prob) == len(self.ser_cat):
                self.ser_prob = ser_prob / np.sum(ser_prob) #归一化概率，到达流量的概率
            else:
                self.ser_prob = np.ones(len(ser_cat)) / len(ser_cat)
        else:
            self.ser_prob = np.array([1])
            self.band_ser_cat = self.band_whole

        self.UE_cat = self.rng.choice(self.ser_cat, self.UE_max_no, p=self.ser_prob) #TBD从三种类型中抽，重复一百次，按照p的概率分布
        self.tx_pkt_no = np.zeros(len(self.ser_cat))      #  到达包不会按照同一种分布吧。

    def channel_model(self,rng):
    #信道模型
        if self.chan_mod == '36814':
            shadowing_var = 8 #rayleigh fading shadowing variance 8dB
            self.chan_loss = self.path_loss + rng.normal(0,shadowing_var,self.UE_max_no).reshape(-1,1)

    def scheduling(self):
    #调度模型，应该是值把每个切片内的资源分配给UE，真正改变的是self.UE_band

        self.UE_band = np.zeros(self.UE_max_no) # initializing
        if self.schedu_method == 'round_robin':
            ser_cat = len(self.ser_cat)
            band_ser_cat = self.band_ser_cat #在外面就根据动作来分配
            if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000): #（5，10，。。。）%10000== 5，就是说每秒刚开始的一步，一共2000个slot
                self.ser_schedu_ind =  [0] * ser_cat #大概是1s里每个时隙都会剩一点 精细分配
                
            for i in range(ser_cat): 
                UE_index = np.where((self.UE_buffer[0,:]!=0) & (self.UE_cat == self.ser_cat[i]))[0] #UE_buffer第一个维度干什么用，挑出来又到达流量的用户
                UE_Active_No = len(UE_index)
                if UE_Active_No != 0:
                    RB_No = band_ser_cat[i] // (180 * 10**3) #180k
                    RB_round = RB_No // UE_Active_No #均匀分给每个用户。这种很明显可以改成比例公平调度器
                    self.UE_band[UE_index] += 180 * 10**3 * RB_round #每个用户分到的带宽

                    RB_rem_no = int(RB_No - RB_round * UE_Active_No) #不能整除，有剩余的带宽
                    left_no = np.where(UE_index > self.ser_schedu_ind[i])[0].size  #这是个什么含义
                    if left_no >= RB_rem_no:     
                        UE_act_index = UE_index[np.where(np.logical_and(np.greater_equal(UE_index,self.ser_schedu_ind[i]),np.less(UE_index, RB_rem_no + self.ser_schedu_ind[i])))]
                        #下面的逻辑是UE_index<RB_rem_no + self.ser_schedu_ind[i]
                        if UE_act_index.size != 0:
                            self.UE_band[UE_act_index] += 180 * 10**3
                            self.ser_schedu_ind[i] = UE_act_index[-1] + 1 
                    else:
                        UE_act_index_par1 = UE_index[np.where(UE_index>self.ser_schedu_ind[i])]
                        UE_act_index_par2 = UE_index[0:RB_rem_no-left_no]
                        self.UE_band[np.hstack((UE_act_index_par1,UE_act_index_par2))] += 180 * 10**3
                        self.ser_schedu_ind[i] = UE_act_index_par2[-1]+1
        elif self.schedu_method == 'round_robin_nons':
            band_whole = self.band_whole
            if self.sys_clock == self.time_subframe:
                self.ser_schedu_ind =  0
                
            UE_index = np.where((self.UE_buffer[0,:]!=0))[0] #不分每个业务了，那这也没有切片的意义啊
            UE_Active_No = len(UE_index)
            if UE_Active_No != 0:
                RB_No = band_whole // (180 * 10**3)
                RB_round = RB_No // UE_Active_No
                
                self.UE_band[UE_index] += 180 * 10**3 * RB_round

                RB_rem_no = RB_No % UE_Active_No
                left_no = np.where(UE_index > self.ser_schedu_ind)[0].size
                if left_no >= RB_rem_no:     
                    UE_act_index = UE_index[np.where(np.logical_and(np.greater_equal(UE_index,self.ser_schedu_ind),np.less(UE_index, RB_rem_no + self.ser_schedu_ind)))]
                    if UE_act_index.size != 0:
                        self.UE_band[UE_act_index] += 180 * 10**3
                        self.ser_schedu_ind = UE_act_index[-1] + 1 
                else:
                    UE_act_index_par1 = UE_index[np.where(UE_index>self.ser_schedu_ind)]
                    UE_act_index_par2 = UE_index[0:RB_rem_no-left_no]
                    self.UE_band[np.hstack((UE_act_index_par1,UE_act_index_par2))] += 180 * 10**3
                    self.ser_schedu_ind = UE_act_index_par2[-1]+1
            if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000):
                self.band_ser_cat = np.zeros(len(self.ser_cat))
            for i in range(len(self.ser_cat)):
                if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000):
                    self.band_ser_cat[i] = np.sum(self.UE_band[self.UE_cat == self.ser_cat[i]])
                else: 
                    self.band_ser_cat[i] += np.sum(self.UE_band[self.UE_cat == self.ser_cat[i]])
                    if (self.sys_clock * 10000) % (self.learning_windows * 10000) == 0:
                        lw = (self.learning_windows * 10000)/(self.time_subframe * 10000)
                        self.band_ser_cat[i] = self.band_ser_cat[i]/lw
    #分配的带宽配置self.succ_tx_pkt_no
    def provisioning(self):
        UE_index = np.where(self.UE_band != 0) 
        self.channel_model(self.rng)
        rx_power = 10 ** ((self.BS_tx_power - self.chan_loss + self.UE_rx_gain)/10) #16-75+20 shape(100,)
        rx_power = rx_power.reshape(1,-1)[0]
        rate = np.zeros(self.UE_max_no)
        rate[UE_index] = self.UE_band[UE_index] * np.log10(1 + rx_power[UE_index] / ( 10 **(self.noise_PSD /10) * self.UE_band[UE_index] )) * self.dl_mimo# 香农信道公式-20.4
        
        buffer = np.sum(self.UE_buffer,axis=0) #shape(100,) 
        UE_index_b = np.where(buffer != 0) #还有待传输数据的用户
        for ue_id in UE_index_b[0]:
            self.UE_latency[:,ue_id]=latencyUpdate(self.UE_latency[:,ue_id],self.UE_buffer[:,ue_id],self.time_subframe)
        
        for ue_id in UE_index[0]: 
            self.UE_buffer[:,ue_id]=bufferUpdate(self.UE_buffer[:,ue_id],rate[ue_id],self.time_subframe)  
        #被改变的主要是self.succ_tx_pkt_no
        self.store_reward(rate)

        self.bufferClear()
        
    #这是流量到达之类的把  主要改的应该是buffer，read_time,以及tx_pkt_no
    def activity(self): #https://www.ngmn.org/fileadmin/user_upload/NGMN_Radio_Access_Performance_Evaluation_Methodology.pdf
        # VoLTE uses the VoIP model
        # embb_general uses the video streaming model
        # urllc uses the FTP2 model
        if self.sys_clock == 0:
            for ser_name in self.ser_cat:
                ue_index = np.where(self.UE_cat == ser_name)
                ue_index_Size = ue_index[0].size
                if ser_name == 'volte':
                    self.UE_readtime[ue_index] = np.random.uniform(0,160 * 10 ** (-3),[1,ue_index_Size]) # the silence lasts 160 ms in maximum
                elif ser_name == 'embb_general':
                    tmp_readtime = self.rng.pareto(1.2,[1,ue_index_Size]) * 6 * 10 ** -3
                    tmp_readtime[tmp_readtime > 12.5 * 10 ** -3] = 12.5 * 10 ** -3
                    self.UE_readtime[ue_index]  = tmp_readtime
                elif ser_name == 'urllc':
                    self.UE_readtime[ue_index]  = self.rng.exponential(180* 10 ** -3,[1,ue_index_Size]) # read time is determines much smaller; the spec shows the average time is 180s, but here it is defined as 180 ms

        for ue_id in range(self.UE_max_no):
            if self.UE_readtime[ue_id] <= 0:
                if self.UE_buffer[:,ue_id].size - np.count_nonzero(self.UE_buffer[:,ue_id]) != 0: # The buffer is not fullnp.count_nonzero()统计数组中非零元素的个数
                    
                    buf_ind = np.where(self.UE_buffer[:,ue_id] == 0)
                    buf_ind = buf_ind[0][0] #这是一行 去此行的第0个
                    if self.UE_cat[ue_id] == 'volte':
                        self.UE_buffer[buf_ind,ue_id] = 40 * 8 #这应该是流量到达320
                        self.UE_readtime[ue_id] = self.rng.uniform(0,160 * 10 ** (-3),1)
                    elif self.UE_cat[ue_id] == 'embb_general':
                        tmp_buffer_size = self.rng.pareto(1.2,1) * 800
                        if tmp_buffer_size > 2000:
                            tmp_buffer_size = 2000
                        # tmp_buffer_size = np.random.choice([1*8*10**6, 2*8*10**6, 3*8*10**6, 4*8*10**6, 5*8*10**6])
                        self.UE_buffer[buf_ind,ue_id] = tmp_buffer_size
                        self.UE_readtime[ue_id] = self.rng.pareto(1.2,[1,1]) * 6 * 10 ** -3
                        if self.UE_readtime[ue_id] > 12.5 * 10 ** -3:
                            self.UE_readtime[ue_id] = 12.5 * 10 ** -3 
                    elif self.UE_cat[ue_id] == 'urllc':
                        #tmp_buffer_size = np.random.lognormal(14.45,0.35,[1,1])
                        # if tmp_buffer_size > 5 * 10 **6:
                        #      tmp_buffer_size > 5 * 10 **6
                        # tmp_buffer_size = np.random.choice([6.4*8*10**3, 12.8*8*10**3, 19.2*8*10**3, 25.6*8*10**3, 32*8*10**3])
                        tmp_buffer_size = self.rng.choice([0.3*8*10**6, 0.4*8*10**6, 0.5*8*10**6, 0.6*8*10**6, 0.7*8*10**6])
                        self.UE_buffer[buf_ind,ue_id] = tmp_buffer_size
                        self.UE_readtime[ue_id]  = self.rng.exponential(180* 10 ** -3,[1,1]) # read time is determines much smaller; the spec shows the average time is 180s, but here it is defined as 180 ms
                    self.tx_pkt_no[self.ser_cat.index(self.UE_cat[ue_id])] += 1
                    self.UE_buffer_backup[buf_ind,ue_id] = self.UE_buffer[buf_ind,ue_id] #成功接收 指的是，先读了，然后在处理完缓存了。
                    
            else:
                self.UE_readtime[ue_id] -= self.time_subframe
        self.sys_clock += self.time_subframe
        self.sys_clock = round(self.sys_clock,4)
       
    def get_state(self):
        #state = np.zeros(len(self.ser_cat))
        #for ser_name in self.ser_cat:
        #    ue_index = np.where(self.UE_cat == ser_name)
        #    state[self.ser_cat.index(ser_name)] = np.where(self.UE_buffer[0,ue_index[0]] != 0)[0].size
        state = self.tx_pkt_no
        return state
        
    def store_reward(self,rate):
    # 计算系统的吞吐率和QoE满足率。
        # Calculating the SE and EE for each UE
        se = np.zeros(len(self.ser_cat))
        ee = np.zeros(len(self.ser_cat))
        sys_rate_frame = 0
        for ser_name in self.ser_cat:
            ser_index = self.ser_cat.index(ser_name)
            ue_index_ = np.where(self.UE_cat == ser_name)
            allo_band = np.sum(self.UE_band[ue_index_])
            sum_rate = np.sum(rate[ue_index_])
            if allo_band != 0:
                sys_rate_frame += sum_rate
                se[ser_index] = sum_rate/allo_band #bits/hz这个是频谱效率啊，其实所有频谱都用上了，信道条件好的才能用上更多的频谱，这个效率就高，但是这个调度没啥影响吧，全靠随机，轮询的那种可能还好点
                ee[ser_index] = se[ser_index]/10**(self.BS_tx_power/10)# 这俩完全是正比关系
        
        # Calculating the system SE and EE
        self.sys_se_per_frame += sys_rate_frame/self.band_whole

        handling_latency = 2 * 10 ** (-3) #处理时延也是固定的，2ms，实际上是0ms
        handling_latency = 0
        for ue_id in range(self.UE_max_no): 
            for i in range(self.UE_latency[:,ue_id].size):#这有点解释为什么是5的意思，规定时延到，算成接收成功的包数
                if (self.UE_buffer[i,ue_id] == 0) & (self.UE_latency[i,ue_id] != 0): 
                    if self.UE_cat[ue_id] == 'volte': 
                        cat_index = self.ser_cat.index('volte')   #0
                        if (self.UE_latency[i,ue_id] == self.time_subframe):
                            if (rate[ue_id] >= 51 * 10 ** 3) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency): 
                                self.succ_tx_pkt_no[cat_index] += 1
                        else:
                            if (self.UE_buffer_backup[i,ue_id]/self.UE_latency[i,ue_id] >= 51 * 10 ** 3) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                    elif self.UE_cat[ue_id] == 'embb_general':
                        cat_index = self.ser_cat.index('embb_general')    
                        if (self.UE_latency[i,ue_id] == self.time_subframe):
                            #if (rate[ue_id] >= 5 * 10 ** 6) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency):
                            if (rate[ue_id] >= 100 * 10 ** 6) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                        else:
                            if (self.UE_buffer_backup[i,ue_id]/self.UE_latency[i,ue_id] >= 100 * 10 ** 6) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                    elif self.UE_cat[ue_id] == 'urllc': 
                        cat_index = self.ser_cat.index('urllc')   
                        if (self.UE_latency[i,ue_id] == self.time_subframe):
                            if (rate[ue_id] >= 10 * 10 ** 6) & (self.UE_latency[i,ue_id] < 3 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                        else:
                            if (self.UE_buffer_backup[i,ue_id]/self.UE_latency[i,ue_id] >= 10 * 10 ** 6) & (self.UE_latency[i,ue_id] < 3 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1

    def get_reward(self):
        se_total = self.sys_se_per_frame/(self.learning_windows / self.time_subframe)
        # ee_total = se_total/10**(self.BS_tx_power/10)   

        reward = self.succ_tx_pkt_no/self.tx_pkt_no 
        
        return reward, se_total

    def bufferClear(self):    
        latency = np.sum(self.UE_latency,axis=0)
        UE_index = np.where(latency != 0) 
        bufSize = self.UE_latency[:,0].size # 5
        for ue_id in UE_index[0]: 
            
            buffer_ = self.UE_buffer[:,ue_id].copy() #shape(5,1) copy是深复制，切片是视图，一个改变都会改变
            buffer_bk = self.UE_buffer_backup[:,ue_id].copy()
            latency_ = self.UE_latency[:,ue_id].copy()
            ind_1 = np.where(np.logical_and(buffer_ ==0 , latency_ !=0 ) )#就说这是传输过的时延有了，buffer传输完了是0
            indSize_1 = ind_1[0].size
            if indSize_1 != 0:
                self.UE_latency[ind_1,ue_id] = np.zeros(indSize_1) #传输过的变为0
                self.UE_buffer_backup[ind_1,ue_id] = np.zeros(indSize_1)# 传输过的变为0

  
            ind = np.where(np.logical_and(buffer_ !=0 , latency_ !=0 ) ) # #，没有传输完，但是传输了一部分
            ind = ind[0]
            indSize = ind.size 
            if indSize != 0:
                self.UE_buffer[:,ue_id] = np.zeros(bufSize)
                self.UE_latency[:,ue_id] = np.zeros(bufSize)
                self.UE_buffer_backup[:,ue_id] = np.zeros(bufSize)
                self.UE_buffer[:indSize,ue_id] = buffer_[ind] #移到前面来，相当于队列的先进先出
                self.UE_latency[:indSize,ue_id] = latency_[ind]
                self.UE_buffer_backup[:indSize,ue_id] = buffer_bk[ind]
            
    def countReset(self):
        self.tx_pkt_no = np.zeros(len(self.ser_cat))
        '''for ser_name in self.ser_cat:
            ser_index = self.ser_cat.index(ser_name)
            ue_index_ = np.where(self.UE_cat == ser_name)
            self.tx_pkt_no[ser_index] = np.where(self.UE_buffer[:,ue_index_]!=0)[0].size'''
        self.succ_tx_pkt_no = np.zeros(len(self.ser_cat)) #成功接收的包的数量
        self.sys_se_per_frame = np.zeros(1)  #系统每帧的频谱效率
        self.UE_buffer = np.zeros(self.UE_buffer.shape) #UE的缓存
        self.UE_buffer_backup = np.zeros(self.UE_buffer.shape)
        self.UE_latency = np.zeros(self.UE_buffer.shape)

          
def bufferUpdate(buffer,rate,time_subframe):    
    bSize = buffer.size #为什么要有5个呢，rate为0，主要每个buffer也不限大小啊，
    for i in range(bSize):
        if buffer[i] >= rate * time_subframe:
            buffer[i] -= rate * time_subframe
            rate = 0
            break
        else:
            rate_ = buffer[i]
            buffer[i] = 0
            rate -= rate_
    return buffer

def latencyUpdate(latency,buffer,time_subframe):
    lSize = latency.size
    for i in range(lSize):
        if buffer[i]!=0:
            latency[i] += time_subframe
    return latency
