import argparse, math, os
import numpy as np
import gym
from gym import wrappers

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.pg_learning import PG_discrate_net


if __name__ == '__main__':
    # use the cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
       print('using the GPU...')
       torch.cuda.manual_seed(3000)#为当前GPU设置随机种子
       torch.cuda.manual_seed_all(3000)#为所有GPU设置随机种子
    else:
       print('using the CPU...')
       torch.manual_seed(3000)

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    # 取0.99 的效果是相对最好的
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--num_steps', type=int, default=3000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=20000, metavar='N',
                        help='number of episodes (default: 2000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--ckpt_freq', type=int, default=100,
                        help='model saving frequency')
    parser.add_argument('--display', type=bool, default=False,
                        help='display or not')
    parser.add_argument("--action_space", default=1, type=int)
    parser.add_argument("--state_size", default=1, type=int)

    args = parser.parse_args()

    env = gym.make(args.env_name)
    args.action_space = env.action_space.n
    args.state_size = env.observation_space.shape[0]

    env.seed(args.seed)
    env = env.unwrapped
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = PG_discrate_net(args, device)
    total_step = []
    for i_episode in range(args.num_episodes):
        state = torch.Tensor([env.reset()])
        entropies = []
        log_probs = []
        rewards = []
        for t in range(args.num_steps):
            action, log_prob, entropy = agent.choose_action2(state)
            action = action.cpu()
            next_state, reward, done, _ = env.step(action.numpy()[0])
            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = torch.Tensor([next_state])

            if done:
                break

        agent.learning(rewards, log_probs, entropies, args.gamma)
        print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
        total_step.append(len(rewards))

    plt.figure(1)
    plt.plot(total_step)
    plt.savefig('./model_save/'+str(args.env_name)+'-step.png')
    env.close()
