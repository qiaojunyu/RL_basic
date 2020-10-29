import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class Actor_discrete(nn.Module):
    def __init__(self, state, hidden, action):
        super(Actor_discrete,self).__init__()
        self.fc1 = nn.Linear(state,hidden)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(hidden, action)
        self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        action_probability = F.softmax(actions_value)
        return action_probability


class Critic_discrete(nn.Module):
    def __init__(self,state,hidden):
        super(Critic_discrete,self).__init__()
        self.fc1 = nn.Linear(state,hidden)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(hidden, 1)
        self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.out(x)
        return value


class Actor_continuous(nn.Module):
    def __init__(self,hidden, state, action):
        super(Actor_continuous,self).__init__()
        self.fc1 = nn.Linear(state,hidden)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(hidden,action)
        self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        means = nn.Tanh(actions_value)
        return means


class Critic_continuous(nn.Module):
    def __init__(self,hidden, state):
        super(Critic_continuous,self).__init__()
        self.fc1 = nn.Linear(state,hidden)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(hidden, 1)
        self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        values = self.fc2(x)
        return values