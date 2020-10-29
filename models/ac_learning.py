import torch

from layer.AC import Actor_discrete, Critic_discrete

import numpy as np

import torch.nn.utils as utils

class AC_discrate_net():

    def __init__(self, arg, device):

        self.model_actor = Actor_discrete(arg.hidden_size, arg.state_size, arg.action_space).to(device)
        self.model_critic = Critic_discrete(arg.hidden_size, arg.state_size, arg.value).to(device)

        # 参数

        self.action_number = arg.action_space
        self.state_size = arg.state_size

        self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr=1e-4)
        self.optimizer_critic = torch.optim.Adam(self.model_critic.parameters(), lr=1e-4)

        self.model_actor.train()
        self.model_critic.train()
        self.device = device

    def select_action(self, state):
        probs = self.model_actor(state.to(self.device))
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        # prob = probs[:, action].view(1, -1)
        # log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()
        return action, log_prob, entropy

    def learning(self, reward, state, new_state, done, log_prob, entropies, gamma):

        #reward = torch.Tensor(reward).to(self.device)
        state = torch.Tensor(state).to(self.device)
        new_state = torch.Tensor(new_state).to(self.device)

        critic_value = self.model_critic(state)
        new_critic_value = self.model_critic(new_state)
        #a2c
        critic_delta = reward + gamma*new_critic_value*(1-int(done)) - critic_value
        actor_loss = -log_prob*critic_delta - 0.01*entropies
        critic_loss = critic_delta**2

        loss = actor_loss + critic_loss

        loss.to(self.device)

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        loss.backward()

        utils.clip_grad_norm(self.model_actor.parameters(), 40)
        utils.clip_grad_norm(self.model_critic.parameters(), 40)

        self.optimizer_actor.step()
        self.optimizer_critic.step()




