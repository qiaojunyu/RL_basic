import argparse, math, os
import numpy as np
import gym
from gym import wrappers
import torch
from models.sac_learning_2 import SAC_gaussian
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # use the cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print('using the GPU...')
        torch.cuda.manual_seed(3000)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(3000)  # 为所有GPU设置随机种子
    else:
        print('using the CPU...')
        torch.manual_seed(3000)

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

    parser.add_argument('--seed', type=int, default=3000)
    #env
    parser.add_argument('--env_name', type=str, default='Pendulum-v0')
    parser.add_argument("--action_space", default=1, type=int)
    parser.add_argument("--state_size", default=1, type=int)
    parser.add_argument("--max_action", default=1, type=int)
    parser.add_argument("--value", default=1, type=int)

    #play
    parser.add_argument('--num_episodes', type=int, default=100000, metavar='N',
                        help='number of episodes (default: 20000)')
    parser.add_argument('--max_length_of_trajectory', type=int, default=200, metavar='N',
                        help='number of episodes (default: 2000)')

    #network
    parser.add_argument('--hidden_size', type=int, default=500, metavar='N',
                        help='number of episodes (default: 400)')


    #replay
    parser.add_argument('--buffer_maxlen', type=int, default=100000, metavar='N',
                        help='replay buffer ')
    parser.add_argument("--batch_size", default=256, type=int)

    # learning
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
                        help='learning rate 0.001)' )
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='parameter update')
    parser.add_argument('--update_iteration', type=int, default=10, metavar='G',
                        help='update_iteration')
    # save
    parser.add_argument('--directory', type=str, default='./model_save/')

    args = parser.parse_args()
    env = gym.make(args.env_name)
    args.action_space = env.action_space.shape[0]
    print("env.action_space.shape",env.action_space.shape)
    args.state_size = env.observation_space.shape[0]
    args.max_action = float((env.action_space.high- env.action_space.low) / 2.0 + \
                 (env.action_space.high + env.action_space.low) / 2.0)
    env.seed(args.seed)
    # env = env.unwrapped
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = SAC_gaussian(args, device)
    total_step = []
    for i_episode in range(args.num_episodes):
        state = env.reset()
        # state = torch.Tensor([env.reset()])
        rewards = 0
        for t in range(args.max_length_of_trajectory):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, next_state, done_mask))
            state = next_state
            rewards += reward
            # print(t)
            if done:
                break
            if agent.buffer.buffer_len() > 500:
                agent.update()
        agent.writer.add_scalar('reward', rewards, i_episode)
        print("episode:{}, reward:{}, buffer_capacity:{}".format(i_episode, rewards, agent.buffer.buffer_len()))

        if i_episode % 30 == 0:
            agent.save()

    for i in range(3000):
        state = env.reset()
        agent.load()
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