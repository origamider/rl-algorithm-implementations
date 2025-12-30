import numpy as np
import gymnasium as gym
import os
import sys
sys.path.append(os.pardir)
from algorithms.dqn import DQNAgent

episodes = 50
sync_interval = 20
env = gym.make('CartPole-v1',render_mode='human')
agent = DQNAgent()
ct = 0
reward_history = []
for episode in range(episodes):
    state,info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state,reward,terminated,truncated,info = env.step(action)
        
        env.render()
        done = terminated or truncated
        agent.update(state,action,reward,next_state,done)
        state = next_state
        total_reward += reward
    ct += 1
    print(ct)
    if episode % sync_interval == 0:
        agent.sync_qnet()
    reward_history.append(total_reward)

env.close()
print(total_reward)