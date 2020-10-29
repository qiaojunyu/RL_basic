import torch.nn as nn

import torch.nn.functional as F

# AC分为第三动作和连续动作
# 离散的


class Actor_discrete(nn.Module):
    def __init__(self,hidden, state, action):
        super(Actor_discrete,self).__init__()
        self.fc1 = nn.Linear(state,hidden)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(hidden,action)
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
    def __init__(self,hidden, state, value):
        super(Critic_discrete,self).__init__()
        self.fc1 = nn.Linear(state, hidden)
        self.fc1.weight.data.normal_(0, 0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0, 0.1) # initialization
        self.out = nn.Linear(hidden, value)
        self.out.weight.data.normal_(0, 0.1) # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class Actor_continuous(nn.Module):
    def __init__(self,hidden, state, action):
        super(Actor_continuous,self).__init__()
        self.fc1 = nn.Linear(state,hidden)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(hidden,action)
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
    def __init__(self,hidden, state, value):
        super(Critic_discrete,self).__init__()
        self.fc1 = nn.Linear(state, hidden)
        self.fc1.weight.data.normal_(0, 0.1) # initialization
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc2.weight.data.normal_(0, 0.1) # initialization
        self.out = nn.Linear(hidden, value)
        self.out.weight.data.normal_(0, 0.1) # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value