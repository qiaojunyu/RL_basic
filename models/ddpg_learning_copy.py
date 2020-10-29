import torch
from layer.DDPG import Actor, Critic
import torch.nn as nn
import numpy as np
import random
from collections import namedtuple
from torch.autograd import Variable
from tensorboardX import SummaryWriter

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


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
        self.action_number = arg.action_space
        self.state_size = arg.state_size
        self.max_action = arg.max_action
        # replay
        self.replay_buffer = Replay_buffer(arg.memory_size)
        # 参数更新
        self.lr = arg.lr
        self.gamma = arg.gamma
        self.learn_step_counter = 0 # for target updating
        self.tau = arg.tau
        self.update_iteration = arg.update_iteration
        self.batch_size = arg.batch_size

        # 梯度

        self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr=1e-4)
        self.optimizer_critic = torch.optim.Adam(self.model_critic.parameters(), lr=1e-3)
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
            x, y, u, r, d = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1-d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # 构造目标参数，detach 表示梯度不更新
            target_Q = self.model_critic_target(next_state, self.model_actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q).detach()
            current_Q = self.model_critic(state, action)
            critic_loss = self.loss_func(current_Q, target_Q)

            self.writer.add_scalar('critic_loss', critic_loss, self.learn_step_counter)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            actor_loss = -self.model_critic(state, self.model_actor(state)).mean()
            self.writer.add_scalar('actor_loss', actor_loss, self.learn_step_counter)
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            for param, target_param in zip(self.model_actor.parameters(), self.model_actor_target.parameters()):
                target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

            for param, target_param in zip(self.model_critic.parameters(), self.model_critic_target.parameters()):
                target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

            self.learn_step_counter += 1


    def save(self):
        torch.save(self.model_actor,self.directory+'actor.pth')
        torch.save(self.model_critic,self.directory+'critic.pth')
        print('model saved')

    def load(self):
        self.model_actor.load_state.dict(torch.load(self.directory+'actor.pth'))
        self.model_critic.load_state.dict(torch.load(self.directory+'critic.pth'))
        print('model load')


