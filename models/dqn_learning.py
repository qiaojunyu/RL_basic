import numpy as np
from layer.DQN import Net
import torch
import torch.nn as nn
from torch.autograd import Variable


class DQN_net(object):
    def __init__(self,arg,device):

        self.eval_net = Net(arg.state_size,arg.action_space).to(device)
        self.target_net = Net(arg.state_size,arg.action_space).to(device)
        # 参数
        self.action_number= arg.action_space
        self.state_size= arg.state_size
        self.memory_size = arg.memory_size
        self.lr = arg.lr
        self.gamma = arg.gamma
        self.target_replaece_net = arg.target_replaece_net
        self.batch_size = arg.batch_size

        self.learn_step_counter = 0 # for target updating
        self.memory_counter = 0 # for storing memory
        self.epsilon = arg.epsilon
        self.epsilon_max = arg.epsilon_max
        self.epsilon_increment = arg.epsilon_increment

        self.memory = np.zeros((self.memory_size,self.state_size*2+2)) # initailize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.device = device

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x),0)).to(self.device)
        if np.random.uniform() < self.epsilon: # greedy
            actions_value = self.eval_net.forward(x)
            action = np.argmax(actions_value.cpu().data.numpy())
        else:
            action = np.random.randint(0,self.action_number)
        return action

    def choose_action_test(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x),0)).to(self.device)
        actions_value = self.eval_net.forward(x)
        action = np.argmax(actions_value.cpu().data.numpy())
        return action

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn_dqn(self):
        # target net update
        if self.learn_step_counter % self.target_replaece_net == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,self.batch_size)

        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        b_memory = self.memory[sample_index,:]
        b_s = torch.FloatTensor(b_memory[:,:self.state_size]).to(self.device)
        b_a = torch.LongTensor(b_memory[:,self.state_size:self.state_size+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:,self.state_size+1:self.state_size+2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:,-self.state_size:]).to(self.device)

        print(b_a.size())
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # 评估网络
        # 目标函数
        # # 目标网络计算出来的值
        # q_next = self.target_net(b_s_).detach() # 不更新过梯度
        # q_target = b_r + self.gamma * q_next.max(1)[0]

        q_next = self.target_net(b_s_).max(1)[0].reshape(self.batch_size,1)
        q_next = q_next.detach()
        q_target = q_next * self.gamma + b_r

        loss = self.loss_func(q_eval,q_target )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max

        self.learn_step_counter = self.learn_step_counter + 1

    def learn_double_dqn(self):
        # target net update
        if self.learn_step_counter % self.target_replaece_net == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,self.batch_size)

        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        b_memory = self.memory[sample_index,:]


        b_s = torch.FloatTensor(b_memory[:,:self.state_size]).to(self.device)
        b_a = torch.LongTensor(b_memory[:,self.state_size:self.state_size+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:,self.state_size+1:self.state_size+2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:,-self.state_size:]).to(self.device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        # 评估网络

           # 目标函数

        # # 目标网络计算出来的值
        # q_next = self.target_net(b_s_).detach() # 不更新过梯度
        # q_target = b_r + self.gamma * q_next.max(1)[0]
        next_state_selected = self.eval_net(b_s_).detach().argmax(1)

        q_next = self.target_net(b_s_).gather(1,next_state_selected.unsqueeze(-1))

        q_next = q_next.detach()
        q_target = q_next * self.gamma + b_r

        loss = self.loss_func(q_eval,q_target )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max

        self.learn_step_counter = self.learn_step_counter + 1

    def save(self):
        torch.save(self.target_net.state_dict(), 'target_net.pth')
        torch.save(self.eval_net.state_dict(),   'model_save\eval_net.pkl')
        #print("====================================")
        #print("Model has been saved...")
        #print("====================================")
    def load(self):
        self.target_net.load_state_dict(torch.load('target_net.pth'))
        self.eval_net.load_state_dict(torch.load('model_save\eval_net.pkl'))





