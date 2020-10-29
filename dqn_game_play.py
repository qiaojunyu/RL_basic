import argparse

import gym
import torch

from models.dqn_learning import DQN_net

from tensorboardX import SummaryWriter

def train(config, device, writer):

    step_list = []
    i_episode_list = []
    env = gym.make(config.env)
    env = env.unwrapped
    config.action_space = env.action_space.n
    config.state_size = env.observation_space.shape[0]

    dqn = DQN_net(config,device)
    step_max = 0
    for i_episode in range(30000):
        s = env.reset()
        step = 0
        ep_r = 0
        while True:
           # env.render()
            a = dqn.choose_action(s)
            # take action
            s_,r,done,info = env.step(a)
            # 修改奖励
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold *0.7
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians *0.3
            r = r1 + r2
            step = step + 1
            if step >100000:
                print("step>100000")
                done = True
            ep_r += r
            dqn.store_transition(s,a,r,s_)
            if dqn.memory_counter > 1000 and dqn.memory_counter % dqn.batch_size == 0:
                     print(dqn.memory_counter)
                     dqn.learn_dqn()
            if done:
                print('episode: ', i_episode,
                  ' ep_r: ', round(ep_r, 2),
                  ' step: ', round(step, 2),
                  ' epsilon: ', round(dqn.epsilon, 2),
                      )
                step_list.append(step)
                i_episode_list.append(i_episode)
                writer.add_scalar("ep_reward", ep_r, i_episode)
                writer.add_scalar("step", step, i_episode)
                writer.add_scalar("epsilon", dqn.epsilon, i_episode)
                if step_max < step:
                     step_max = step
                     dqn.save()
                     print('episode: ', i_episode, 'model update and save:', "max_step:",step_max)
                break
            s = s_
    # plt.plot(i_episode_list,step_list)
    # plt.show()



def estimation(config, device):

      env = gym.make(config.env)
      env = env.unwrapped # b
      config.action_space = env.action_space.n
      config.state_size = env.observation_space.shape[0]
      print('\n test')
      dqn1 = DQN_net(config, device)
      dqn1.load()
      for i_episode in range(200):
        s = env.reset()
        total = 0
        step = 0
        while True:
            #env.render()
            a = dqn1.choose_action_test(s)
            # take action
            s_,r,done,info = env.step(a)
            total = total + r
            step = step + 1
            if step >100000:
                print("step>10000")
                done = True
            if done:
                print("total,i_episode",total,i_episode)
                break
            s = s_


if __name__ == '__main__':


    # torch.manual_seed(1)    # reproducible

    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        default='CartPole-v0',
                        help="Name of environment")

    parser.add_argument("--action_space", default=1, type=int)
    parser.add_argument("--state_size", default=1, type=int)
    parser.add_argument("--memory_size", default=10000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--target_replaece_net", default=30, type=float)
    parser.add_argument("--batch_size", default= 300, type=int)
    parser.add_argument("--epsilon", default=0.4, type=float)
    parser.add_argument("--epsilon_max", default=0.98, type=float)
    parser.add_argument("--epsilon_increment", default=0.001, type=float)

    config = parser.parse_args()
    # use the cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
       print('using the GPU...')
       torch.cuda.manual_seed(3000)#为当前GPU设置随机种子
       torch.cuda.manual_seed_all(3000)#为所有GPU设置随机种子
    else:
       print('using the CPU...')
       torch.manual_seed(3000)


    writer = SummaryWriter(comment="-" + config.env)

    train(config,device,writer)
    estimation(config,device)

