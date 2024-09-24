# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:48:52 2024

@author: MMH_user
"""
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from prepare_data import prepare_data
from LSTM_Torch_CUDA import *

#https://www.geeksforgeeks.org/long-short-term-memory-networks-using-pytorch/

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#check:
#device = 'cpu'

###############################################################################
#0) creating the data set
t_start = -50
t_end   = 10
incr    = 0.15

t       = np.arange(t_start,t_end, incr)
t       = t.reshape(len(t),1)
Y_t     = np.sin(t) + 0.1*np.random.randn(len(t),1) + np.exp((t + 20)*0.05)

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

[X, Y] = prepare_data(Y_tnorm, dt_past, 1)

cut            = int(np.round(0.6*Y_tnorm.shape[0]))

#reshape to fit input requirements for torch LSTM
X = X.reshape((X.shape[0], X.shape[1]))
Y = Y.reshape((Y.shape[0]))

TrainX, TrainY = X[:cut], Y[:cut]
TestX,   TestY = X[cut:], Y[cut:]

TrainX = torch.tensor(TrainX[:, :, None], dtype=torch.float32)
TrainY = torch.tensor(TrainY[:, None], dtype=torch.float32)

TestX  = torch.tensor(TestX[:, :, None], dtype=torch.float32)
TestY  = torch.tensor(TestY[:, None], dtype=torch.float32)

TrainX = TrainX.to(device)
TrainY = TrainY.to(device)

TestX = TestX.to(device)
TestY = TestY.to(device)


torch.cuda.synchronize()

###############################################################################
#training model 
n_epochs  = 100
n_neurons = 200
n_stack   = 2 #number of stacked LSTMs
dt_futu   = 5

model     = LSTMModel(input_dim = n_features, hidden_dim = n_neurons,\
                      layer_dim = n_stack, output_dim = dt_futu,\
                      device = device)
model     = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

t1 = time.monotonic()
# Training loop
for epoch in range(n_epochs):
    
    outputs = model(TrainX)
    optimizer.zero_grad()
    loss = criterion(outputs, TrainY)
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))

torch.cuda.synchronize()
    
# Predicted outputs
PredY = model(TestX).detach().to('cpu').numpy()

t2 = time.monotonic()
dt = t2 -t1

print('runtime is ' + str(dt))

back  = PredY.shape[0]

plt.plot(t, Y_tnorm, linewidth = 5)
plt.plot(t[-back:],PredY[:,0])
plt.legend(['actual data', 'prediction'])
plt.fill_between([t[-back,0], t[-1,0]],\
                  0, 1, color = 'k', alpha = 0.1)
plt.plot([t[-back,0], t[-back,0]], [0, 1],'k-',linewidth = 3)
plt.show()








