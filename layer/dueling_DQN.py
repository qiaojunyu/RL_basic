import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim
import numpy as np

class Dueling_DQN(nn.Module):
    def __init__(self, int_dims, hidden, actions):
        super(Dueling_DQN,self).__init__()
        self.fc1 = nn.Linear(int_dims, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.V = nn.Linear(hidden, 1)
        self.A = nn.Linear(hidden, actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        V = self.V(x)
        A = self.A(x)
        return V, A   # 此处的V是指状态值函数， A是指每个动作值函数的优势函数