import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim


class Descrete_policy(nn.Module):
    def __init__(self, hidden_size, observation, action_number):
        super(Descrete_policy, self).__init__()
        self.action_space = action_number

        self.linear1 = nn.Linear(observation, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_number)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        action_value = self.linear2(x)
        action_probability = F.softmax(action_value)
        return action_probability



