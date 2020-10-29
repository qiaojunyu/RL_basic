import gym
import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal

from models.sac_new_learning import  SAC

def main(env, agent, Episode, batch_size):
    Return = []
    for episode in range(Episode):
        score = 0
        state = env.reset()
        for i in range(3000):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, next_state, done_mask))
            state = next_state
            score += reward
            if done:
                break
            if agent.buffer.buffer_len() > 500:
                agent.update(batch_size)

        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent.buffer.buffer_len()))
        Return.append(score)
        score = 0
    env.close()
    plt.plot(Return)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    env = env.unwrapped
    # Params
    tau = 0.005
    gamma = 0.99
    q_lr = 3e-3
    value_lr = 3e-3
    policy_lr = 3e-3
    buffer_maxlen = 80000

    Episode = 500
    batch_size = 400

    agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, device)
    main(env, agent, Episode, batch_size)