import numpy as np
import gymnasium as gym
import torch
import os
import sys
sys.path.append(os.pardir)
from algorithms.dqn import DQNAgent

episodes = 10
sync_interval = 20
env = gym.make('CartPole-v1',render_mode='human')
agent = DQNAgent()
qnet_path = 'dqn_cartpole_final.pth'
if os.path.exists(qnet_path):
    agent.qnet.load_state_dict(torch.load(qnet_path))
    print('Successfully loaded the training model!')
ct = 0
reward_history = []
print("Training DQN agent...")

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

torch.save(agent.qnet.state_dict(),'dqn_cartpole_final.pth')
print("Training completed!")
env.close()
print(total_reward)