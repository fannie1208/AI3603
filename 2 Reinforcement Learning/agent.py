# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import random
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, epsilon=0.9, alpha=0.5, gamma=0.9, num_space=48):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_space = num_space
        self.Q_list = [{} for _ in range(self.num_space)]

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        action = np.random.choice(self.all_actions)
        if random.random() < self.epsilon:
            return action
        max_q = float('-inf')
        for a, q in self.Q_list[observation].items():
            if q > max_q:
                action, max_q = a, q
        return action
    
    def learn(self, s, s_, a, a_, reward):
        """learn from experience"""
        self.Q_list[s][a] = (1-self.alpha)*self.Q_list[s].get(a,0) + self.alpha*(reward+self.gamma*self.Q_list[s_].get(a_,0))
        # time.sleep(0.5)
        # print("What I should learn? (ﾉ｀⊿´)ﾉ")
        return True
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, epsilon=0.9, alpha=0.5, gamma=0.9, num_space=48):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_space = num_space
        self.Q_list = [{} for _ in range(self.num_space)]

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        action = np.random.choice(self.all_actions)
        if random.random() < self.epsilon:
            return action
        max_q = float('-inf')
        for a, q in self.Q_list[observation].items():
            if q > max_q:
                action, max_q = a, q
        return action
    
    def learn(self, s, s_, action, reward):
        """learn from experience"""
        max_q = float('-inf') if self.Q_list[s_] else 0
        for _, q in self.Q_list[s_].items():
            max_q = max(max_q, q)
        self.Q_list[s][action] = (1-self.alpha)*self.Q_list[s].get(action,0) + self.alpha*(reward+self.gamma*max_q)
        print(self.Q_list[s][action])
        # time.sleep(0.5)
        # print("What I should learn? (ﾉ｀⊿´)ﾉ")
        return True
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####
