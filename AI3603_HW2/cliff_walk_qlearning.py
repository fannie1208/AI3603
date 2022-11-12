# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import QLearningAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import argparse
import os.path as osp
from datetime import datetime
from utils.logger import Logger
from utils.helper import args_print
##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 1
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

##### START CODING HERE #####
def parse_args(): 
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--e", type=float, default=1, help="the initialized value of epsilon")
    parser.add_argument("--alpha", type=float, default=1, help="the initialized value of alpha")
    parser.add_argument("--gamma", type=float, default=0.9, help="the discount factor gamma")
    parser.add_argument("--de", type=float, default=0.99, help="the decreasing rate of epsilon")
    parser.add_argument("--da", type=float, default=0.99, help="the decreasing rate of alpha")
    parser.add_argument('--result_path', type=str, default='results/qlearning/', help='Path where the result will be saved in.')
    args = parser.parse_args()
    return args

args = parse_args()

# log
datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name = f'qlearn.alpha_{args.alpha}.gamma_{args.gamma}.epsilon_{args.e}.de_{args.de}.da_{args.da}.seed_{args.seed}.{datetime_now}'
exp_dir = osp.join(f'{args.result_path}', experiment_name)
os.makedirs(exp_dir, exist_ok=True)
logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
args_print(args, logger)

# construct the intelligent agent.
agent = QLearningAgent(all_actions,epsilon=args.e, alpha=args.alpha, gamma=args.gamma)

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        # env.render()
        # update the episode reward
        episode_reward += r
        print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience
        agent.learn(s,s_,a,r)
        s = s_
        if isdone:
            agent.Q_list[s][a] = r
            # time.sleep(0.1)
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)
    logger.info(f"==Episode: {episode} --episode_reward: {episode_reward} --epsilon: {agent.epsilon}")
    logger.info(f"=====QlearningTable: {agent.Q_list}")

    agent.alpha *= args.da
    agent.epsilon *= args.de 
print('\ntraining over\n')   

# reset for visualization
agent.epsilon = 0
logger.info(f"\n=========== The Final Path ===========")
# reset env
s = env.reset()
# render env
env.render()
while True:
    # choose an action
    a = agent.choose_action(s)
    logger.info(f"state: {s}; action: {a}")
    s_, r, isdone, info = env.step(a)
    time.sleep(0.5)
    env.render()
    s = s_
    if isdone:
        break

# close the render window after training.
env.close()

##### END CODING HERE #####


