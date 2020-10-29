import argparse, math, os
import numpy as np
import gym
from gym import wrappers
import torch
from models.ddpg_learning import DdpgNet
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

    parser.add_argument('--is_surpvised', default=True, type=bool)
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
    # 1e-3 效果比较好
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
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
    env = env.unwrapped
    agent = DdpgNet(args, device)
    total_step = []
    for i_episode in range(args.num_episodes):
        state = env.reset()
        state = torch.Tensor([state])
        # state = torch.Tensor([env.reset()])
        rewards = 0
        for t in range(args.max_length_of_trajectory):
          #  env.render()
            action = agent.select_action(state)
            # 增加噪声,np.random.normal,表示正态函数，其中0表示均值，中间的表示方差，最后表示输出的值的个数
            action = (action + np.random.normal(0, args.exploration_noise, size = env.action_space.shape[0]))
            action = action.clip(env.action_space.low, env.action_space.high)
            next_state, reward, done, info = env.step(action)
            rewards = rewards + reward

            action = torch.Tensor(action)
            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            agent.store_transition_temporary(state, action, mask, next_state, reward)
            # agent.store_transition(state, action, mask, next_state, reward)
            state = next_state
            if done:
                break
            # 状态变化
            # if t%500 == 0:
            #    print("Episode: {}, step:{}, reward: {}".format(i_episode,t,rewards))
        #if i_episode % args.ckpt_freq == 0:
        #torch.save(agent.model.state_dict(), 'reinforce-' + str(i_episode) + '.pkl')
        if i_episode == 0:
            agent.reward_max = rewards

        if agent.memory_counter > 1000:
                agent.learning()
        if args.is_surpvised and i_episode >100:
            # 采用监督学习模式，选取较好的样本进行更新
            if rewards > agent.reward_max:
                print("agent.reward_max,", rewards)
                agent.reward_max = rewards
                agent.from_temporary_to_memory()
                agent.memory_temporary = []
            # 如果是负值，需要对回报进行压缩，主要是负值
            elif abs(rewards) < abs(agent.reward_max+200) and rewards < 0:
                 print("rewards:{}, max_reward:{}, memory insert".format(rewards,agent.reward_max))
                 agent.from_temporary_to_memory()
                 agent.memory_temporary = []
            # 如果回报是正值，则选取范围内小10%的数据
            elif abs(rewards) > abs(agent.reward_max-200) and rewards > 0:
                 print("rewards:{}, max_reward:{}, memory insert".format(rewards,agent.reward_max))
                 agent.from_temporary_to_memory()
                 agent.memory_temporary = []
            # 如果这个轨迹不好，直接删除
            else:
                 agent.memory_temporary = []
        else:
             agent.from_temporary_to_memory()
             agent.memory_temporary = []

        agent.writer.add_scalar('reward', rewards, i_episode)
        print("Episode: {}, reward: {}".format(i_episode, rewards))
        total_step.append(t)

        if i_episode % 500 == 0:
            agent.save()
        if i_episode % 100 == 0:
            plt.figure(1)
            plt.plot(total_step)
            plt.savefig('./model_save/' + str(args.env_name) + '-step.png')

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
