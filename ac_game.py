import argparse, math, os
import numpy as np
import gym
from gym import wrappers
import torch
from models.ac_learning import AC_discrate_net
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

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
    parser.add_argument('--num_steps', type=int, default=10000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=30000, metavar='N',
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
    parser.add_argument("--value", default=1, type=int)



args = parser.parse_args()

env = gym.make(args.env_name)
args.action_space = env.action_space.n
args.state_size = env.observation_space.shape[0]

env.seed(args.seed)
env = env.unwrapped
torch.manual_seed(args.seed)
np.random.seed(args.seed)

writer = SummaryWriter(comment="-" + args.env_name)

agent = AC_discrate_net(args, device)
total_step = []
for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()])
    rewards = 0
    for t in range(args.num_steps):
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()
        next_state, reward, done, _ = env.step(action.numpy()[0])
        rewards = rewards + reward
        agent.learning(reward, state, next_state, done, log_prob, entropy, args.gamma)
        if done:
            break
        # 状态变化
        state = torch.Tensor([next_state])


    #if i_episode % args.ckpt_freq == 0:
    #torch.save(agent.model.state_dict(), 'reinforce-' + str(i_episode) + '.pkl')
    writer.add_scalar("ep_reward", rewards, i_episode)
    writer.add_scalar("step",t+1 , i_episode)
    print("Episode: {}, step:{}, reward: {}".format(i_episode,t+1, rewards))
    total_step.append(t)

plt.figure(1)
plt.plot(total_step)
plt.savefig('./model_save/'+str(args.env_name)+'-step.png')
env.close()

