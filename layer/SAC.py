import torch
import torch.nn as nn
import torch.nn.functional as F


class SAC_actor_gaussian(nn.Module):
    def __init__(self, state, hidden, num_actions, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(SAC_actor_gaussian,self).__init__()
        self.fc1 = nn.Linear(state, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden, num_actions)
        self.mu_head.weight.data.uniform_(-edge, edge)
        self.mu_head.bias.data.uniform_(-edge, edge)

        self.log_std_head = nn.Linear(hidden, num_actions)
        self.log_std_head.weight.data.uniform_(-edge, edge)
        self.log_std_head.bias.data.uniform_(-edge, edge)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
         # action rescaling

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x) # 千万不用relu
        log_std_head = self.log_std_head(x) # 千万不用relu,否则就是娶不到小于0
        # 压缩取值范围到固定的范围
        log_std_head = torch.clamp(log_std_head, self.log_std_min, self.log_std_max)
        return mu, log_std_head


class SAC_critic_v(nn.Module):
    def __init__(self, state, hidden_dim, edge=3e-3):
        super(SAC_critic_v, self).__init__()
        self.linear1 = nn.Linear(state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # 初始化网络参数
        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SAC_critic_q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, edge=3e-3):
        super(SAC_critic_q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # 初始化网络参数
        self.fc3.weight.data.uniform_(-edge, edge)
        self.fc3.bias.data.uniform_(-edge, edge)

    def forward(self, s, a):
        # s = s.reshape(-1, state_dim)
        # a = a.reshape(-1, action_dim)
        x = torch.cat([s, a], 1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


