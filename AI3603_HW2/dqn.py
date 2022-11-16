# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
# Add my log files
import os.path as osp
from datetime import datetime
from utils.logger import Logger
from utils.helper import args_print

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=1000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=100,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=0.3,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """comments: """
    """
    Create the Neural Network, which is composed by linear layers and activation layers.
    The activate function is ReLU.
    Use 'model = QNetwork(env)' to initialize the network, and 'model(x)' for forward propagation.
    The class inherits the 'nn.Module' class
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """comments: """
    """
    This function calculates the epsilon during the training process.
    'duration' is the steps it takes from start_e to end_e.
    epsilon(t) = max(start_e + slope * t, end_e) and epsilon(duration) = end_e
    However, I think this function calculates the slope repeatedly, which is a waste of time.
    I think the slope can be calculated at the beginning of the training, and the function can only return max(slope * t + start_e, end_e)
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

reward = []
if __name__ == "__main__":

    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'dqn.lr_{args.learning_rate}.gamma_{args.gamma}.e_{args.start_e}.ende_{args.end_e}.buffer_{args.buffer_size}.tnf_{args.target_network_frequency}.bs{args.batch_size}.seed_{args.seed}.{datetime_now}'
    exp_dir = osp.join('results/dqn/', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
    args_print(args, logger)

    """we utilize tensorboard yo log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    """comments: """
    """
    set random seed and make the result reproducible.
    The initial weight parameters in deep learning network models are usually initialized to random numbers.
    The selection of random seeds can reduce the randomness of algorithm results to a certain extent, that is,
    setting random seeds means that the random numbers generated are the same every time the experiment is run.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """comments: """
    """
    construct the gym environment:
    env = gym.make(env_id) # construct the environment of "LunarLander-v2"
    env = gym.wrappers.RecordEpisodeStatistics(env) # wrap the gym environment to record episode statistics
    env.seed(seed) # set random seed to make the result reproducible
    env.action_space.seed(seed) # set random seed to make the result reproducible
    env.observation_space.seed(seed) # set random seed to make the result reproducible
    """
    envs = make_env(args.env_id, args.seed)

    """comments: """
    """
    Initialize the Q-Network and target network.
    The parameters of Target network and Q-network are separated.
    Only the parameters of Q-network will be updated. And Q's parameters are copied to Target-network periodically.
    The optimizer chooses Adam optimizer.
    """
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """comments: """
    """
    The sampled samples are dropped into ReplayBuffer.
    When the number of samples reaches a certain threshold, an update will be made, using the data in ReplayBuffer.
    There're some benefits of the use of Replay Buffer:
    1. Eliminate the correlation of sampling data
    2. Better stabilize the data distribution
    """
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """comments: """
    """
    Reset env at the beginning of the training.
    """
    obs = envs.reset()
    for global_step in range(args.total_timesteps):

        """comments: """
        """
        utilize the linear_schedule function to set the value of epsilon every epoch.
        """
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        """comments: """
        """
        implement the epsilon-greedy method.
        if random float is less than epsilon, the agent will take action randomly.
        Otherwise, it'll choose the action with maximal q_value.
        """
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()

        """comments: """
        """
        make the agent take a step using the chosen actions and get the necessary information
        """
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training

        if dones:
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
            logger.info(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            reward.append(float(infos["episode"]["r"]))

        """comments: """
        """
        Add the sample data to the replay buffer in order to update the agent later.
        """
        rb.add(obs, next_obs, actions, rewards, dones, infos)

        """comments: """
        """
        If it's done, reset the env. Otherwise, repeat the procedures
        (assign next_obs to obs in order to get the next actions).
        """
        obs = next_obs if not dones else envs.reset()

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:

            """comments: """
            """
            sample the data from Replay Buffer to calculate the loss and backward the network
            """
            data = rb.sample(args.batch_size)

            """comments: """
            """
            1. with torch.no_grad():
            indicates that the current calculation doesn't need backpropagation
            and the following content is forced not to build the calculation graph.
            2. Use MSE Loss to calculate the loss. loss(x,y)=(x-y)^2
            3. L(\theta) = E[Q'-Q]^2 = E[r + gamma*target_max - Q]^2
            """
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """comments: """
            """
            Record the td_loss and q_values to the tensorboard every 100 steps.
            """
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

            """comments: """
            """
            optimizer.zero_grad(): clear the past gradient
            loss.backward(): back propagate, calculate the current gradient
            optimizer.step(): update the parameters of the network according to the gradient
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """comments: """
            """
            Every time the agent learns 'target_network_frequency' timesteps, the target network will be updated.
            """
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
    torch.save(q_network.state_dict(), f"dqn.lr_{args.learning_rate}.gamma_{args.gamma}.e_{args.start_e}.ende_{args.end_e}.buffer_{args.buffer_size}.tnf_{args.target_network_frequency}.bs{args.batch_size}.seed_{args.seed}.{datetime_now}.pth")
    average_reward = sum(reward[-20:])/20
    logger.info(f"The average reward of last 20 steps: {average_reward}")
    """close the env and tensorboard logger"""
    # show the result of DQN
    # obs = envs.reset()
    # envs.render()
    # while True:
    #     q_values = q_network(torch.Tensor(obs).to(device))
    #     actions = torch.argmax(q_values, dim=0).cpu().numpy()
    #     next_obs, rewards, dones, infos = envs.step(actions)
    #     envs.render()
    #     if dones:
    #         break
    #     obs = next_obs
    envs.close()
    writer.close()