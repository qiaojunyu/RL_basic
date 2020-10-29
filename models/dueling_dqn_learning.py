import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim
import numpy as np
from layer.dueling_DQN import Dueling_DQN

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_counter = 0
        self.memory_state = np.zeros(self.mem_size, *input_shape, dtype =np.float32)
        self.memory_next_state = np.zeros(self.mem_size, *input_shape, dtype =np.float32)
        self.memory_action = np.zeros(self.mem_size, dtype =np.int64)
        self.memory_reward = np.zeros(self.mem_size, dtype =np.float32)
        self.memory_done = np.zeros(self.mem_size, dtype =np.uint8)

    def store_trainsition(self,state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size
        self.memory_state[index] = state
        self.memory_next_state[index] = next_state
        self.memory_action[index] = action
        self.memory_reward[index] = reward
        self.memory_done[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        # 参数replace 用来设置是否可以取相同元素： True表示可以取相同数字；False表示不可以取相同数字
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.memory_state[batch]
        actions = self.memory_action[batch]
        next_states = self.memory_next_state[batch]
        rewards = self.memory_reward[batch]
        dones = self.memory_done[batch]
        return states, actions, rewards, next_states, dones

class dueling_DQN():
     def __init__(self, arg, device):

         self.dueling_dqn_online = Dueling_DQN(arg.state_size, arg.hidden_size, arg.action_space).to(device)
         self.dueling_dqn_target = Dueling_DQN(arg.state_size, arg.hidden_size, arg.action_space).to(device)

         self.optimizer = optim.Adam(self.dueling_dqn_online.parameters(), lr=self.lr )
         self.loss = nn.MSELoss()
         self.directory = arg.directory
         self.replay_buffer = ReplayBuffer(arg.mem_size,arg.state_size)
         self.epsilon  = arg.epsilon

     def choose_action(self,state):
         if np.random.random() > self.epsilon:
             pass



