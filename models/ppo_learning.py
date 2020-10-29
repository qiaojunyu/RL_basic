import torch
from layer.PPO import Actor_continuous, Critic_continuous, Actor_discrete,Critic_discrete
import torch.nn as nn
import numpy as np
import random
from collections import namedtuple
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.utils as utils
from tensorboardX import SummaryWriter

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PpoDiscrete:

    def __init__(self, arg, device):
         # evel, 目标网络与采样网络
        self.model_actor = Actor_discrete(arg.state_size, arg.hidden_size, arg.action_space).to(device)
        self.model_actor_sample = Actor_discrete(arg.state_size, arg.hidden_size, arg.action_space).to(device)
        self.model_actor_sample.load_state_dict(self.model_actor.state_dict())

        self.model_critic = Critic_discrete(arg.state_size, arg.hidden_size).to(device)

          # 环境参数
        self.action_number = arg.action_space
        self.state_size = arg.state_size
        self.model_actor.train()
        self.model_critic.train()

        # replay
        self.gamma = arg.gamma
        self.memory = Memory() # initailize memory
        # 参数更新
        self.update_iteration = arg.update_iteration
        self.lr = arg.lr
        self.eps_clip = arg.eps_clip

        # 梯度
        self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr=self.lr )
        self.optimizer_critic = torch.optim.Adam(self.model_critic.parameters(), lr=self.lr )
        self.loss_func = nn.MSELoss()
        self.device = device

        # 模型保存
        self.writer = SummaryWriter(comment="-" + arg.env_name)
        self.directory = arg.directory
        self.writer = SummaryWriter(comment="-" + arg.env_name)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(self.device)
        # or  state = torch.from_numpy(state).float().to(device)
        # state.reshape(1,-1) 表示转化为一行
        action_probs = self.model_actor_sample(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
       # action_to_cpu = action.cpu().data.numpy().flatten()
        action_to_cpu = action.item()
        return action_to_cpu, action, action_logprob

    def store_transition(self, state, action, action_logprob, reward, done):
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob)
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

    # 新的策略，评估一下差距
    def evaluate(self, state, action):
        action_probs = self.model_actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.model_critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def learning(self):
         # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs), 1).to(self.device).detach()

        # Optimize policy for K epochs:

         # 这里是两个网络,多次训练，然后更新，第一次的交叉熵基本是1，因为网络是一致的
        for ep in range(self.update_iteration):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
             #PPO2
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss,优势函数, 需要固定梯度
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.loss_func(state_values, rewards) - 0.1*dist_entropy
            # print(loss,loss.size())
            # take gradient step
            loss.to(self.device)

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.mean().backward()
            utils.clip_grad_norm(self.model_actor.parameters(), 40)
            utils.clip_grad_norm(self.model_critic.parameters(), 40)
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        # Copy new weights into old policy:
        self.memory.clear_memory()
        self.model_actor_sample.load_state_dict(self.model_actor.state_dict())

    def save(self):
        torch.save(self.model_actor,self.directory+'actor.pth')
        torch.save(self.model_critic,self.directory+'critic.pth')
        print('model saved')

    def load(self):
        self.model_actor.load_state.dict(torch.load(self.directory+'actor.pth'))
        self.model_critic.load_state.dict(torch.load(self.directory+'critic.pth'))
        print('model load')



