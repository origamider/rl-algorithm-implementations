import sys
import os
sys.path.append(os.pardir)

import gymnasium as gym
import numpy
from algorithms.monte_carlo import MonteCarloAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
env = gym.make("FrozenLake-v1",is_slippery=False)
agent = MonteCarloAgent(env)
reward_data = []
epsilon_data = []

def train(epsilon_decay=0.999,last_epsilon=0.01,num_episodes=5000):
    for episode in tqdm(range(num_episodes)):
        obs,info = env.reset()
        agent.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.add(obs,action,reward)
            obs = next_obs
            done = terminated or truncated
            total_reward += reward
        agent.update()
        agent.epsilon = max(last_epsilon,agent.epsilon*epsilon_decay)
        epsilon_data.append(agent.epsilon)
        reward_data.append(total_reward)

def visualize(num_episodes=10):
    agent.epsilon = 0.0
    render_env = env = gym.make("FrozenLake-v1",render_mode="human",is_slippery=False)
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        agent.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            done = terminated or truncated
    render_env.close()

train()
env.close()
plt.plot(epsilon_data)
plt.show()
visualize()