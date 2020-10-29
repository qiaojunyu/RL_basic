from layer.pg_discrete import Descrete_policy
import torch
import torch.nn.utils as utils
import numpy as np

class PG_discrate_net():

    def __init__(self, arg, device):

        self.model = Descrete_policy(arg.hidden_size, arg.state_size, arg.action_space).to(device)
        # 参数

        self.action_number = arg.action_space
        self.state_size = arg.state_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.model.train()
        self.device = device

    def select_action(self, state):
        probs = self.model(state.to(self.device))
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        # prob = probs[:, action].view(1, -1)
        # log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):

        R = torch.zeros(1, 1)
        loss = 0
        # reversed 函数返回一个反转的迭代器。
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i] # 每一步的累积回报
            loss = loss - (log_probs[i] * (R.expand_as(log_probs[i])).to(self.device)).sum() - \
                   (0.1 * entropies[i].to(self.device)).sum()
        loss = loss / len(rewards)
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()

    def choose_action2(self, state):
        probs = self.model(state.to(self.device))
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        entropy = - (probs*probs.log()).sum()

        return action, log_probs, entropy

    def learning(self, rewards, log_probs, entropies, gamma):
        G = np.zeros_like(rewards, dtype = np.float64)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum = G_sum + rewards[k]*discount
                discount = discount*gamma
            G[t] = G_sum
        # 归一化，正则化,提升变化率
        mean = np.mean(G)
        if np.std(G) > 0:
            std = np.std(G)
        else:
            std = 1
        G = (G-mean)/std

        G = torch.Tensor(G).to(self.device)
        loss = 0
        for R, logprob, entropy in zip(G, log_probs, entropies):
            loss = loss - R*logprob - 0.1* entropy

        loss.to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()











