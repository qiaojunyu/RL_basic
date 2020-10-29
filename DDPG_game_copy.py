import argparse, math, os
import numpy as np
import gym
from gym import wrappers
import torch
from models.ddpg_learning_copy import DdpgNet
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # use the cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print('using the GPU...')
    else:
        print('using the CPU...')

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--random_seed', type=int, default=9527)
    parser.add_argument('--seed', default=False, type=bool)
    #env
    parser.add_argument('--env_name', type=str, default='Pendulum-v0')

    parser.add_argument("--action_space", default=1, type=int)
    #network
    parser.add_argument('--hidden_size', type=int, default=400, metavar='N',
                        help='number of episodes (default: 400)')
    parser.add_argument("--state_size", default=1, type=int)
    parser.add_argument("--max_action", default=1, type=int)

    #play
    parser.add_argument('--num_episodes', type=int, default=100000, metavar='N',
                        help='number of episodes (default: 20000)')
    parser.add_argument('--max_length_of_trajectory', type=int, default=3000, metavar='N',
                        help='number of episodes (default: 2000)')

    #replay
    parser.add_argument('--memory_size', type=int, default=100000, metavar='N',
                        help='replay buffer ')
    parser.add_argument("--batch_size", default=100, type=int)

    # learning
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    # 学习率在1e-4的时候不收敛， 1e-3会收敛
    parser.add_argument('--lr', type=float, default=1e-2, metavar='G',
                        help='learning rate 0.001)' )

    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='parameter update')

    parser.add_argument('--update_iteration', type=int, default=200, metavar='G',
                        help='update_iteration')

    parser.add_argument('--exploration_noise', type=float, default=0.1, metavar='G',
                        help='exploration_noise')
    # save
    parser.add_argument('--directory', type=str, default='./model_save/')

    args = parser.parse_args()
    env = gym.make(args.env_name)
    args.action_space = env.action_space.shape[0]
    args.state_size = env.observation_space.shape[0]
    args.max_action = float(env.action_space.high[0])

    if args.seed:
       env.seed(args.random_seed)
       torch.manual_seed(args.random_seed)
       np.random.seed(args.random_seed)
    # env = env.unwrapped
    agent = DdpgNet(args, device)
    total_step = []
    for i_episode in range(args.num_episodes):
        state = env.reset()
        rewards = 0
        for t in range(args.max_length_of_trajectory):
            action = agent.select_action(state)
            # 增加噪声,np.random.normal,表示正态函数，其中0表示均值，中间的表示方差，最后表示输出的值的个数
            action = (action + np.random.normal(0, args.exploration_noise, size = env.action_space.shape[0]))
            action = action.clip(env.action_space.low, env.action_space.high)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
            rewards = rewards + reward
            state = next_state
            if done:
                break
        agent.learning()
        print(" Episode: \t{} Total Reward: \t{:0.2f}".format(i_episode, rewards))

    for i in range(3000):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reawrd, done, info = env.step(np.float32(action))
            ep_r += reward
            env.render()
            if done or t>=args.max_length_of_trajector:
                agent.writer.add_scalar('test_reward', ep_r, i)
                print('Episode:{}, Return:{:0.2f}, Step:{}'.format(i,ep_r,t))
                ep_r = 0
                break
            state = next_state
    env.close()
