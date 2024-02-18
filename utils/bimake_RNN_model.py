#!/urs/bin/env python3

"""
Create and update recurrent neural network.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, LSTM,Bidirectional,Conv1D,Dropout,GRU,MaxPooling1D,TimeDistributed
#from tensorflow.keras.utils import np_utils
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import History
from tensorflow.keras import optimizers
from tensorflow_core.python.keras.engine.sequential import Sequential

from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply
from keras.models import Model


from attention_utils import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

import  pandas as pd
import  numpy as np


def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def make_RNN_model(window_size, i_x, added_params=0):
    """
    Create RNN model

    Parameters
    ----------
    window_size : RNN window size as integer
    i_x : number of grid points as integer网格数
    added_params : number of additional parameters as integer (optional)附加的变量数（可选择的）
    Returns：结果
    -------
    model : keras model
    创建RNN模型参数-window_size:RNN窗口大小为整数
    i_x:网格点数为整数
    added _ params:附加参数数量为整数(可选)
    keras模型
    """

    # Define model architecture
    #model = Sequential()

    inputs = Input(shape=(window_size, i_x+added_params))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # , padding = 'same'
    #x = Dropout(0.3)(x)
    #x1= MaxPooling1D(pool_size=2)(x)
    attention_mul = attention_3d_block2(x)
    attention_mul = Flatten()(attention_mul)
    # model.add(Flatten())
    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # 对于GPU可以使用CuDNNLSTM
    lstm_out = Bidirectional(LSTM(100, return_sequences=False, activation='relu'))(attention_mul)#return_sequences=True
    #lstm_out = Dropout(0.3)(lstm_out)
    attention_mul1 = attention_3d_block2(lstm_out)
    attention_mul1 = Flatten()(attention_mul1)
    Dense1 = Dense(200, activation='relu')(attention_mul1)
    Dense2 = Dense(200, activation='relu')(Dense1)
    output = Dense(i_x, activation='sigmoid')(Dense2)
    model = Model(inputs=[inputs], outputs=output)


    # Compile model编译模型，使用RMSprop优化器
    optimizer = optimizers.RMSprop(lr=1e-4, rho=0.9)#RMSprop
    #optimizer = optimizers.Adam(lr=lr1, beta_1=0.9, beta_2=0.999)
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate)
    loss = 'mean_squared_error'
    #loss= 'root_mean_squared_error'
    #loss = 'mean_absolute_error'
    #loss='huber_loss'
    #loss='binary_crossentropy'
    #loss='mean_squared_logarithmic_error'
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mse', 'mae'])

    return model


def update_RNN_model(model):
    """
    Update RNN model: learning rate, loss, etc..
    更新RNN模型:学习率、损失等..
    Parameters
    ----------
    model : keras model

    Returns
    -------
    model : keras model
    """

    # Compile model
    optimizer = optimizers.RMSprop(lr=1e-5)
    loss = 'mean_squared_error'
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mse', 'mae'])

    return model
