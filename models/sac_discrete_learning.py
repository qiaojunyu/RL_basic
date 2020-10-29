import torch
import random
import collections
import torch.optim as optim
from layer.SAC_discrete import Dueling_DQN, Actor_discrete
from tensorboardX import SummaryWriter
import os
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F

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
        state_list = torch.squeeze(torch.stack(state_list).to(self.device), 1)
        action_list = torch.squeeze(torch.stack(action_list).to(self.device), 1)
        reward_list = torch.squeeze(torch.stack(reward_list).to(self.device), 1)
        next_state_list = torch.squeeze(torch.stack(next_state_list).to(self.device), 1)
        done_list = torch.squeeze(torch.stack(done_list).to(self.device), 1)
        # return torch.FloatTensor(state_list).to(self.device), \
        #        torch.LongTensor(action_list).unsqueeze(-1).to(self.device), \
        #        torch.FloatTensor(reward_list).unsqueeze(-1).to(self.device), \
        #        torch.FloatTensor(next_state_list).to(self.device), \
        #        torch.FloatTensor(done_list).unsqueeze(-1).to(self.device)
        return state_list, action_list, reward_list, next_state_list, done_list

    def buffer_len(self):
        return len(self.buffer)

class SAC_discrete():
    def __init__(self, arg, device):
         # evel, 目标网络与采样网络
             # 参数更新
        self.gamma = arg.gamma
        self.update_iteration = arg.update_iteration
        self.lr = arg.lr
        self.learn_step_counter = 0 # for target updating
        self.tau = arg.tau
          # 环境参数
        self.action_size = arg.action_space
        self.state_size = arg.state_size
        # replay
        self.batch_size = arg.batch_size
        # Initialize thebuffer
        self.buffer = ReplayBeffer(arg.memory_size, device)
        self.device = device

        self.actor_local = Actor_discrete(arg.state_size, arg.hidden_size, arg.action_space).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),lr=self.lr)

        self.critic_local = Dueling_DQN(arg.state_size, arg.hidden_size, arg.action_space).to(device)
        self.critic_local_2 = Dueling_DQN(arg.state_size, arg.hidden_size, arg.action_space).to(device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=self.lr)

        self.critic_target = Dueling_DQN(arg.state_size, arg.hidden_size, arg.action_space).to(device)
        self.critic_target_2 = Dueling_DQN(arg.state_size, arg.hidden_size, arg.action_space).to(device)

         # Load the target value network parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        # Load the target value network parameters
        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_local_2.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


        # alpha
        self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr = self.lr, eps=1e-4)


        # 模型保存
        self.writer = SummaryWriter(comment="-" + arg.env_name)
        self.directory = arg.directory

        os.makedirs('./SAC/SAC_model/', exist_ok=True)

    # def select_action(self, state):
        # state = torch.FloatTensor(state).to(self.device)
        # probs = self.actor_local(state)
        # # 采样
        # action_distribution = Categorical(probs)
        # action = action_distribution.sample()
        # action_to_cpu = action.detach().cpu().numpy()
        # return action, action_to_cpu

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(self.device)
        # or  state = torch.from_numpy(state).float().to(device)
        # state.reshape(1,-1) 表示转化为一行
        action_probs = self.actor_local(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_to_cpu = action.item()
        return action, action_to_cpu,

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

   # 用于test 模型
    def act(self, states):
        action_logits = self.policy_net(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1, keepdim=True)
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
        action_batch = action_batch.view(-1,1)
        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action_probabilities, log_action_probabilities= self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        entropies = -torch.sum(
            action_probabilities * log_action_probabilities, dim=1, keepdim=True)

        q = torch.sum(min_qf_pi * action_probabilities, dim=1, keepdim=True)
        policy_loss = (- q - self.alpha * entropies).mean()
        return policy_loss, entropies.detach()

    def calculate_entropy_tuning_loss(self, entropies):
       assert not entropies.requires_grad
       entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies))
       return entropy_loss

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        for i in range(self.update_iteration):
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.buffer.sample(self.batch_size)

            qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
            policy_loss, entropies = self.calculate_actor_loss(state_batch)

            alpha_loss = self.calculate_entropy_tuning_loss(entropies)
             # Update Soft q
            self.critic_optimizer.zero_grad()
            self.critic_optimizer_2.zero_grad()
            qf1_loss.backward()
            qf2_loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer_2.step()

            # Update Policy
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            # updata alpha
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            print("self.alpha", self.alpha)

            for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.critic_target_2.parameters(), self.critic_local_2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            self.learn_step_counter += 1


