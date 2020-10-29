
import torch
import torch.nn as nn

import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, state, hidden, action_space, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state,hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden,action_space)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        res = self.max_action*torch.tanh(x)
        return res


class Critic(nn.Module):
    def __init__(self, state, hidden, action):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state+action, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.out(x)
        return  value

