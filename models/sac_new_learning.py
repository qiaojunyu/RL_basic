import gym
import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal

from layer.SAC_new import ValueNet, SoftQNet, PolicyNet


class ReplayBeffer():
    def __init__(self, buffer_maxlen, device):
        self.buffer = collections.deque(maxlen=buffer_maxlen)
        self.device = device

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return torch.FloatTensor(state_list).to(self.device), \
               torch.FloatTensor(action_list).to(self.device), \
               torch.FloatTensor(reward_list).unsqueeze(-1).to(self.device), \
               torch.FloatTensor(next_state_list).to(self.device), \
               torch.FloatTensor(done_list).unsqueeze(-1).to(self.device)

    def buffer_len(self):
        return len(self.buffer)


class SAC:
    def __init__(self, env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, device):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_range = [env.action_space.low, env.action_space.high]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # initialize networks
        self.value_net = ValueNet(self.state_dim).to(device)
        self.target_value_net = ValueNet(self.state_dim).to(device)
        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim, device).to(device)

        self.action_max = (env.action_space.high- env.action_space.low) / 2.0 + \
                 (env.action_space.high + env.action_space.low) / 2.0

        # Load the target value network parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            # Initialize the optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # Initialize thebuffer
        self.buffer = ReplayBeffer(buffer_maxlen,device)

    def get_action(self, state):
        action = self.policy_net.action(state)
        action = action * (self.action_range[1] - self.action_range[0]) / 2.0 + \
                 (self.action_range[1] + self.action_range[0]) / 2.0

        return action

    def get_evaluate(self, state):
        action,log_prob = self.policy_net.evaluate(state)
        action_max = torch.tensor(self.action_max).to( self.device)
        action = action*action_max
        # action = action.item().clip(env.action_space.low, env.action_space.high)
        # action = torch.tensor(action)
        return action,log_prob
    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        # new_action, log_prob = self.policy_net.evaluate(state)
        new_action, log_prob = self.get_evaluate(state)
        # print("new_action",new_action)
        # print("action",action)
        # V value loss
        value = self.value_net(state)
        new_q1_value = self.q1_net(state, new_action)
        new_q2_value = self.q2_net(state, new_action)
        next_value = torch.min(new_q1_value, new_q2_value) - log_prob
        value_loss = F.mse_loss(value, next_value.detach())

        # Soft q  loss
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = self.target_value_net(next_state)
        target_q_value = reward + done * self.gamma * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

        # Policy loss
        new_q_value = torch.min(new_q1_value, new_q2_value)
        policy_loss = (log_prob - new_q_value).mean()

        # Update v
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward()
        q2_value_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
