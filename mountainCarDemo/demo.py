import torch
import numpy as np
import torch.nn as nn
from collections import deque
import random
import gymnasium as gym
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os

#参考URL
# https://github.com/openai/spinningup/blob/master/spinup/utils/plot.py
# Deep Reinforcement Learning with Double Q-learning(paper)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--load',type=str,default=None,help='指定した.pthファイルを読み込む')
    parser.add_argument('--')

class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self,obs,action,reward,next_obs,done):
        self.buffer.append((obs,action,reward,next_obs,done))
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self):
        data = random.sample(self.buffer,self.batch_size)
        obs = torch.tensor(np.stack([x[0] for x in data]),dtype=torch.float32)
        action =torch.tensor(np.stack([x[1] for x in data]),dtype=torch.long).unsqueeze(dim=1)
        reward = torch.tensor(np.stack([x[2] for x in data]),dtype=torch.float32).unsqueeze(dim=1)
        next_obs = torch.tensor(np.stack([x[3] for x in data]),dtype=torch.float32)
        done = torch.tensor(np.stack([x[4] for x in data]),dtype=torch.float32).unsqueeze(dim=1)
        return obs, action, reward, next_obs, done

class QNet(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3 = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

class BaseAgent:
    def __init__(self,input_dim,output_dim,gamma=0.99,epsilon=1.0):
        self.q = QNet(input_dim,32,output_dim)
        self.q_target = QNet(input_dim,32,output_dim)
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q.parameters())
    
    def get_action(self,obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.output_dim)
        else:
            return torch.argmax(self.q(torch.tensor(obs,dtype=torch.float32))).item()

class AgentWithDoubleDQN(BaseAgent):
    def update(self,obs,action,reward,next_obs,done):
        with torch.no_grad():
            qs = self.q_target(next_obs)
            next_action = torch.argmax(self.q(next_obs),dim=1,keepdim=True) # keepdim=True指定で、(batch_size,1)で固定。つけないと(batch_size,)
            target_q = torch.gather(input=qs,dim=1,index=next_action)
            target = reward + self.gamma*target_q*(1-done)
        
        base_qs = self.q(obs)
        base_q = torch.gather(input=base_qs,dim=1,index=action)
        loss = self.criterion(target,base_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return base_q.mean().item()

class AgentWithDQN(BaseAgent):
    def update(self,obs,action,reward,next_obs,done):
        with torch.no_grad():
            qs = self.q_target(next_obs)
            target_q,_ = torch.max(input=qs,dim=1,keepdim=True)
            target = reward + self.gamma*target_q*(1-done)
        
        base_qs = self.q(obs)
        base_q = torch.gather(input=base_qs,dim=1,index=action)
        loss = self.criterion(target,base_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return base_q.mean().item()

start_epsilon = 1.0
end_epsilon = 0.01
epsilon_decay_steps = 200000
num_episodes = 1000
sync_interval = 10000


def get_energy(obs):
    x = obs[0]
    v = obs[1]
    g = 0.0025
    c = 1/(g+0.07*0.07*0.5)
    res = c*(g*np.sin(3*x)+v*v*0.5)
    return res

def train(agent_type="DQN",num_episodes=500):
    env = gym.make("MountainCar-v0")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = AgentWithDQN(input_dim,output_dim) if agent_type=="DQN" else AgentWithDoubleDQN(input_dim,output_dim)
    replay_buffer = ReplayBuffer(buffer_size=10000,batch_size=32)
    reward_history,q_value_history = [],[]
    total_steps = 0
    for episode in tqdm(range(num_episodes)):
        obs,info = env.reset()
        done = False
        
        total_reward = 0
        while not done:
            total_steps += 1
            action = agent.get_action(obs)
            next_obs, reward, terminated, trunctated, info = env.step(action)
            reward = get_energy(next_obs)-get_energy(obs)
            total_reward += reward
            done = terminated or trunctated
            replay_buffer.add(obs,action,reward,next_obs,done)
            
            if len(replay_buffer) < replay_buffer.batch_size:
                continue
            
            sampled_obs, sampled_action, sampled_reward, sampled_next_obs, sampled_done = replay_buffer.sample()
            average_q = agent.update(sampled_obs,sampled_action,sampled_reward,sampled_next_obs,sampled_done)
            q_value_history.append(average_q)
            
            obs = next_obs
            
            agent.epsilon = end_epsilon + (start_epsilon-end_epsilon)*max(0,(epsilon_decay_steps-total_steps)/epsilon_decay_steps)
            
            if total_steps % sync_interval == 0:
                agent.q_target.load_state_dict(agent.q.state_dict())
        reward_history.append(total_reward)
    env.close()
    return reward_history,q_value_history,agent

dqn_rewards,dqn_qs,dqn_agent = train("DQN")
ddqn_rewards,ddqn_qs,ddqn_agent = train("DoubleDQN")


def smooth_data(data,window=50):
    if len(data) < window:
        return data
    return np.convolve(data,np.ones(window)/window,mode='valid')

fig, ax = plt.subplots(1,2)
ax[0].plot(smooth_data(dqn_rewards),label="DQN Rewards")
ax[0].plot(smooth_data(ddqn_rewards),label="DoubleDQN Rewards")
ax[0].set_title("Reward History")
ax[0].legend()
ax[1].plot(smooth_data(dqn_qs),label="DQN Q-Values")
ax[1].plot(smooth_data(ddqn_qs),label="DoubleDQN Q-Values")
ax[1].set_title("Estimated Q-Values History")
ax[1].legend()
plt.show()

print("--Double DQN Agent Test--")
env = gym.make("MountainCar-v0",render_mode="human")
ddqn_agent.epsilon = 0
for episode in tqdm(range(10)):
    obs,info = env.reset()
    done = False
    
    while not done:
        action = ddqn_agent.get_action(obs)
        next_obs, reward, terminated, trunctated, info = env.step(action)
        done = terminated or trunctated
        obs = next_obs

env.close()