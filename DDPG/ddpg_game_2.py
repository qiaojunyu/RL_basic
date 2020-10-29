import argparse
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
from models.DDPG_learning_2 import DDPG

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
    # OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
    # Note that DDPG is feasible about hyper-parameters.
    # You should fine-tuning if you change to another environment.
    parser.add_argument("--env_name", default="Pendulum-v0")
    parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
    parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
    parser.add_argument('--update_iteration', default=200, type=int)
    parser.add_argument('--target_update_interval', default=1, type=int)

    parser.add_argument('--learning_rate', default=1e-4, type=float)

    parser.add_argument('--memory_size', default=1000000, type=int) # replay buffer size
    parser.add_argument('--batch_size', default=100, type=int) # mini batch size

    parser.add_argument('--test_iteration', default=10, type=int)

    parser.add_argument('--state_dim', default=10, type=int)
    parser.add_argument('--hidden_dim', default=400, type=int)
    parser.add_argument('--action_dim', default=1, type=int)
    parser.add_argument('--max_action', default=2, type=float)


    parser.add_argument('--seed', default=False, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)
    # optional parameters

    parser.add_argument('--sample_frequency', default=2000, type=int)
    parser.add_argument('--render', default=False, type=bool) # show UI or not
    parser.add_argument('--log_interval', default=50, type=int) #
    parser.add_argument('--load', default=False, type=bool) # load model
    parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--max_episode', default=100000, type=int) # num of games
    parser.add_argument('--print_log', default=5, type=int)

    parser.add_argument('--directory', default='directory', type=str) # mode = 'train' or 'test'
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("use ", device)
    script_name = os.path.basename(__file__)
    env = gym.make(args.env_name)

    if args.seed:
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    directory = './model_save/exp' + script_name + args.env_name +'./'
    args.directory = directory


    agent = DDPG(args, device)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.load: agent.load()
        total_step = 0
        for i in range(args.max_episode):
            total_reward = 0
            step =0
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

                next_state, reward, done, info = env.step(action)
                if args.render and i >= args.render_interval:
                    env.render()
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                state = next_state
                if done:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()
           # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")