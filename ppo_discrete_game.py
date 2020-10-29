import argparse, math, os
import numpy as np
import gym
import torch
from models.ppo_learning import PpoDiscrete
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
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument("--action_space", default=1, type=int)
    parser.add_argument("--state_size", default=1, type=int)
    parser.add_argument("--value", default=1, type=int)

    #play
    parser.add_argument('--num_episodes', type=int, default=50000, metavar='N',
                        help='number of episodes (default: 20000)')
    parser.add_argument('--max_length_of_trajectory', type=int, default=3000, metavar='N',
                        help='number of episodes (default: 2000)')

    #network
    parser.add_argument('--hidden_size', type=int, default=400, metavar='N',
                        help='number of episodes (default: 400)')

    # learning
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                        help='learning rate 0.001)' )
    parser.add_argument('--update_iteration', type=int, default=5, metavar='G',
                        help='update_iteration')
    parser.add_argument('--eps_clip', type=int, default=0.2, metavar='G',
                        help='eps_clip')
    parser.add_argument('--update_timestep', type=int, default=200, metavar='G',
                        help='update_timestep')
    # save
    parser.add_argument('--directory', type=str, default='./model_save/')

    args = parser.parse_args()
    env = gym.make(args.env_name)
    args.action_space = env.action_space.n
    args.state_size = env.observation_space.shape[0]
    env.seed(args.seed)
    env = env.unwrapped
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = PpoDiscrete(args, device)
# training loop
    running_reward = 0
    avg_length = 0
    timestep = 0
    for i_episode in range(1, args.num_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(args.max_length_of_trajectory):
            timestep += 1
            # Running policy_old:
            action_to_cpu, action, action_logprob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_to_cpu)
            state = torch.FloatTensor(state.reshape(1,-1))
            agent.store_transition(state, action, action_logprob, reward, done)
            # update if its time
            if timestep % args.update_timestep == 0:
                agent.learning()
                timestep = 0
            running_reward += reward
            # if True:
            #     env.render()
            if done:
                break
            state = next_state
        avg_length += t
        agent.writer.add_scalar('reward', running_reward, i_episode)
        print("Episode: {}, reward: {}".format(i_episode, running_reward))

        # # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(agent.model_actor.state_dict(), './PPO_actor_{}.pth'.format(args.env_name))
        #     torch.save(agent.model_critic.state_dict(), './PPO_critic_{}.pth'.format(args.env_name))
        #     break
        #
        # # logging
        # if i_episode % log_interval == 0:
        #     avg_length = int(avg_length/log_interval)
        #     running_reward = int((running_reward/log_interval))
        #
        #     print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        #     running_reward = 0
        #     avg_length = 0



