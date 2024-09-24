# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:43:06 2024

@author: MMH_user
"""
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,\
                            batch_first = True)
        # Fully connected layer
        fc = nn.Linear(hidden_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.layer_dim  = layer_dim
        
        self.lstm       = lstm.to(device)
        self.fc         = fc.to(device)
        self.device     = device

    def forward(self, x):
        
        device = self.device
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.to(device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = c0.to(device)
        # Detach the gradients to prevent backpropagation through time
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Reshaping the outputs for the fully connected layer
        out = self.fc(out[:, -1, :])
        
        return out