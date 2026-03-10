import os
import sys
sys.path.append(os.pardir)
import numpy
from tqdm import tqdm
import gymnasium as gym
from algorithms.dqn import DQNAgent
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1",render_mode="rgb_array")
agent = DQNAgent(env)
tau = 0.005
reward_data = []
epsilon_data = []

def train(num_episodes=1000,last_episilon=0.01,epsilon_decay_rate=0.99):
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            agent.update(obs,action,reward,next_obs,done)
            obs = next_obs
        agent.epsilon = max(last_episilon,agent.epsilon*epsilon_decay_rate)
        epsilon_data.append(agent.epsilon)
        for param,target_param in zip(agent.q_net.parameters(),agent.q_net_target.parameters()):
            target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
        reward_data.append(total_reward)
    env.close()

def test(num_episodes=10):
    env = gym.make("CartPole-v1",render_mode="human")
    agent.epsilon = 0
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
    env.close()

train()
plt.plot(epsilon_data)
plt.show()
test()