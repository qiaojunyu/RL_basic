import torch
import random
import collections
import torch.optim as optim
from layer.SAC import SAC_actor_gaussian, SAC_critic_v, SAC_critic_q
from tensorboardX import SummaryWriter
import os
import torch.nn.functional as F
from torch.distributions import Normal

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


class SAC_gaussian:
    def __init__(self, arg, device):
         # evel, 目标网络与采样网络
        self.policy_net = SAC_actor_gaussian(arg.state_size, arg.hidden_size, arg.action_space).to(device)

        self.value_net = SAC_critic_v(arg.state_size, arg.hidden_size).to(device)
        self.target_value_net = SAC_critic_v(arg.state_size, arg.hidden_size).to(device)

        self.q1_net = SAC_critic_q(arg.state_size, arg.action_space, arg.hidden_size).to(device)
        self.q2_net = SAC_critic_q(arg.state_size, arg.action_space, arg.hidden_size).to(device)

        # 环境参数
        self.action_number = arg.action_space
        self.state_size = arg.state_size

        self.max_action = arg.max_action

        # 参数更新
        self.batch_size= arg.batch_size
        self.gamma = arg.gamma
        self.update_iteration = arg.update_iteration
        self.lr = arg.lr
        self.learn_step_counter = 0 # for target updating
        self.tau = arg.tau
        #高斯策略采用
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        # 梯度
         # Load the target value network parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            # Initialize the optimizer
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr )
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=self.lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=self.lr)

        self.device = device
        self.min_Val = torch.tensor(1e-7).float().to(device)

        # 模型保存
        self.writer = SummaryWriter(comment="-" + arg.env_name)
        self.directory = arg.directory

        os.makedirs('./SAC_model/', exist_ok=True)

        # Initialize thebuffer
        self.buffer = ReplayBeffer(arg.buffer_maxlen, device)

    # def select_action(self, state):
    #     state = torch.FloatTensor(state).to(self.device)
    #     mu, log_sigma = self.policy_net(state)
    #     sigma = torch.exp(log_sigma)
    #     dist = Normal(mu, sigma)
    #     z = dist.sample()
    #     action = self.max_action*torch.tanh(z)
    #     return action # return a scalar, float32

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.policy_net(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action

    def get_action(self, state):
        action = self.select_action(state)
        action = action*self.max_action
        return action

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.policy_net(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)
        z = noise.sample()
        action = torch.tanh(mean + std * z.to(self.device))
        log_prob = normal.log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob

    def get_evaluate(self, state):
        action,log_prob = self.evaluate(state)
        action_max = torch.tensor(self.max_action).to(self.device)
        action = action*action_max
        return action, log_prob

    def update(self):
        for i in range(self.update_iteration):
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
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

            self.writer.add_scalar('Loss/Q1_loss', q1_value_loss, global_step=self.learn_step_counter)
            self.writer.add_scalar('Loss/Q2_loss', q2_value_loss, global_step=self.learn_step_counter)
            self.writer.add_scalar('Loss/policy_loss', policy_loss, global_step=self.learn_step_counter)
            self.writer.add_scalar('Loss/value_loss', value_loss, global_step=self.learn_step_counter)

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

            self.learn_step_counter += 1

    def save(self):
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.q1_net.state_dict(), './SAC_model/Q_net1.pth')
        torch.save(self.q2_net.state_dict(), './SAC_model/Q_net2.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
        self.value_net.load_state_dict(torch.load( './SAC_model/value_net.pth'))
        self.q1_net.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        self.q2_net.load_state_dict(torch.load('./SAC_model/Q_net2.pth'))
        print("model has been load")