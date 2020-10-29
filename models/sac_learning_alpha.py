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
        self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(self.device)).item() # heuristic value from the paper
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)


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