# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1.0

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        action = np.random.choice(self.all_actions)
        return action
    
    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("What I should learn? (ﾉ｀⊿´)ﾉ")
        return False
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1.0

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        action = np.random.choice(self.all_actions)
        return action
    
    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("What I should learn? (ﾉ｀⊿´)ﾉ")
        return False
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####
