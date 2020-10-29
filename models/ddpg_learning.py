import torch
from layer.DDPG import Actor, Critic
import torch.nn as nn
import numpy as np
import random
from collections import namedtuple
from torch.autograd import Variable
from tensorboardX import SummaryWriter

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))



class DdpgNet():
    def __init__(self, arg, device):

        # evel
        self.model_actor = Actor(arg.state_size, arg.hidden_size, arg.action_space,  arg.max_action).to(device)
        self.model_critic = Critic(arg.state_size, arg.hidden_size, arg.action_space).to(device)

        # target
        self.model_actor_target = Actor( arg.state_size, arg.hidden_size, arg.action_space,  arg.max_action).to(device)
        self.model_critic_target = Critic(arg.state_size, arg.hidden_size, arg.action_space).to(device)

        # 初始参数
        self.model_actor_target.load_state_dict(self.model_actor.state_dict())
        self.model_critic_target.load_state_dict(self.model_critic.state_dict())

        # 环境参数
        # replay
        self.memory_size = arg.memory_size
        self.lr = arg.lr
        self.gamma = arg.gamma
        self.batch_size = arg.batch_size
        self.memory_counter = 0 # for storing memory
        self.memory = [] # initailize memory
        self.memory_temporary = [] # initailize memory

        # 参数更新
        self.learn_step_counter = 0 # for target updating
        self.tau = arg.tau
        self.update_iteration = arg.update_iteration
        self.reward_max = 0

        # 梯度

        self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr=self.lr)
        self.optimizer_critic = torch.optim.Adam(self.model_critic.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.device = device

        # 模型保存

        self.directory = arg.directory
        self.writer = SummaryWriter(comment="-" + arg.env_name)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(self.device)
        # state.reshape(1,-1) 表示转化为一行
        action = self.model_actor(state)
        action_to_cpu = action.cpu().data.numpy().flatten()
        return action_to_cpu

    def learning(self):
        # target net update，软更新
        for i in range(self.update_iteration):
            batch = self.sample()
            state_batch = Variable(torch.cat(batch.state)).to(self.device)
            action_batch = Variable(torch.cat(batch.action)).to(self.device)
            reward_batch = Variable(torch.cat(batch.reward)).to(self.device)
            mask_batch = Variable(torch.cat(batch.mask)).to(self.device)
            next_state_batch = Variable(torch.cat(batch.next_state)).to(self.device)

            reward_batch = reward_batch.unsqueeze(1)
            mask_batch = mask_batch.unsqueeze(1)
            action_batch = action_batch.unsqueeze(1)

            # 构造目标参数，detach 表示梯度不更新
            target_Q = self.model_critic_target(next_state_batch, self.model_actor_target(next_state_batch))
            target_Q = reward_batch +(mask_batch*self.gamma*target_Q).detach()
            current_Q = self.model_critic(state_batch, action_batch)

            critic_loss = self.loss_func(current_Q, target_Q)

            self.writer.add_scalar('critic_loss', critic_loss, self.learn_step_counter)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            actor_loss = -self.model_critic(state_batch, self.model_actor(state_batch)).mean()
            self.writer.add_scalar('actor_loss', actor_loss, self.learn_step_counter)
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            for param, target_param in zip(self.model_actor.parameters(), self.model_actor_target.parameters()):
                target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

            for param, target_param in zip(self.model_critic.parameters(), self.model_critic_target.parameters()):
                target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

            self.learn_step_counter += 1

    def store_transition(self,  *args):
        """Saves a transition."""
        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        index = self.memory_counter % self.memory_size
        self.memory[index] = Transition(*args)
        self.memory_counter += 1

    def store_transition_temporary(self, *args):
        self.memory_temporary.append(Transition(*args))

    def from_temporary_to_memory(self):
        for i in range(len(self.memory_temporary)):
            self.store_transition(self.memory_temporary[i])

    def store_transition(self, transformation):
        """Saves a transition."""
        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        index = self.memory_counter % self.memory_size
        self.memory[index] = transformation
        self.memory_counter += 1

    def sample(self):
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def save(self):
        torch.save(self.model_actor,self.directory+'actor.pth')
        torch.save(self.model_critic,self.directory+'critic.pth')
        print('model saved')

    def load(self):
        self.model_actor.load_state.dict(torch.load(self.directory+'actor.pth'))
        self.model_critic.load_state.dict(torch.load(self.directory+'critic.pth'))
        print('model load')
