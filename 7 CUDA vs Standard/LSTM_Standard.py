# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:00:28 2024

@author: MMH_user
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from prepare_data import prepare_data

###############################################################################
#0) creatting the data set

t_start = -50
t_end   = 10
incr    = 0.15

t   = np.arange(t_start,t_end, incr)
t   = t.reshape(len(t),1)
Y_t = np.sin(t) + 0.1*np.random.randn(len(t),1) + np.exp((t + 20)*0.05)

plt.plot(t, Y_t)
plt.title('complete series')
plt.show()
###############################################################################

###############################################################################
#1) preparing the data set

#1a) scaling
scaler  = MinMaxScaler(feature_range=(0, 1))
Y_tnorm = scaler.fit_transform(Y_t)

#1b) dividing data into training and Test set
n_features = 1
dt_past    = 20
dt_futu    = 5
[X, Y] = prepare_data(Y_tnorm, dt_past, dt_futu)

cut            = int(np.round(0.8*Y_tnorm.shape[0]))

TrainX, TrainY = X[:cut], Y[:cut]
TestX,   TestY = X[cut:], Y[cut:]



#note: the data must have the shape [samples, timesteps, features]
TrainX = TrainX.reshape((TrainX.shape[0], dt_past, n_features))

###############################################################################
#2) creating LSTM
n_neurons  = 200
batch_size = 32


model = Sequential()
model.add(LSTM(n_neurons, input_shape= (dt_past, n_features),\
               activation ='tanh', return_sequences = True))
model.add(LSTM(n_neurons, activation ='relu'))
model.add(Dense(dt_futu))

opt = optimizers.SGD(learning_rate = 0.01, momentum = 0.7)
model.compile(loss = 'mean_squared_error', optimizer = opt)

model.summary()
###############################################################################

###############################################################################
#3) run fit 
n_epochs = 100
t1 = time.monotonic()
out = model.fit(TrainX, TrainY, epochs = n_epochs,\
                    batch_size = batch_size,\
                    validation_split = 0.1, #validation_data = (ValX, ValY),\
                    verbose = 2, shuffle = False)

###############################################################################
#4) prediction
#TestX[0,:,0] should predict TestY[0,0,0]
#TestX[1,:,0] should predict TestY[1,0,0] etc

PredY = model.predict(TestX)

t2 = time.monotonic()
dt = t2 -t1

print('runtime is ' + str(dt))

back  = PredY.shape[0]

plt.plot(t, Y_tnorm, linewidth = 5)
#plt.plot(t[-back:],TestY[:,0,0])
plt.plot(t[-back:],PredY[:,0])
plt.legend(['actual data', 'prediction'])
plt.fill_between([t[-back,0], t[-1,0]],\
                  0, 1, color = 'k', alpha = 0.1)
plt.plot([t[-back,0], t[-back,0]], [0, 1],'k-',linewidth = 3)
plt.show()








