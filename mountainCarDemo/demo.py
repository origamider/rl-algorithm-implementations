import gymnasium as gym
import numpy as np
from  collections import defaultdict
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from collections import deque
import random
import copy
import matplotlib.pyplot as plt

# 参考資料
# https://qiita.com/payanotty/items/07fb38a44cc3bd13e4dd


class QModel(nn.Module):
    # input:obs
    # output:q_function for each action
    def __init__(self,n_input,n_hidden,n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input,n_hidden)
        self.l2 = nn.Linear(n_hidden,n_hidden)
        self.l3 = nn.Linear(n_hidden,n_output)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        val = self.l1(x)
        val = self.relu(val)
        val = self.l2(val)
        val = self.relu(val)
        val = self.l3(val)
        return val

class ReplayBuffer:
    def __init__(self,batch_size,buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self,obs,action,reward,next_obs,done):
        self.buffer.append((obs,action,reward,next_obs,done))
    
    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer,self.batch_size)
        obs = np.stack([x[0] for x in data])
        action = np.stack([x[1] for x in data])
        reward = np.stack([x[2] for x in data])
        next_obs = np.stack([x[3] for x in data])
        done = np.stack([x[4] for x in data])
        
        return obs,action,reward,next_obs,done
    
    
class AgentWithDQN:
    def __init__(self,gamma,epsilon,action_dim,state_dim,device,buffer_size,batch_size):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(self.batch_size,self.buffer_size)
        self.q_net = QModel(state_dim,128,action_dim)
        self.q_net_target = QModel(state_dim,128,action_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters())
    
    def sync_qnet(self):
        self.q_net_target = copy.deepcopy(self.q_net)
    
    def update(self, obs, action, reward, next_obs, done):
        self.optimizer.zero_grad()
        self.replay_buffer.add(obs,action,reward,next_obs,done)
        if len(self.replay_buffer) < self.batch_size:
            return
        
        obs, action, reward, next_obs, done = self.replay_buffer.get_batch()
        
        reward = torch.tensor(reward).float()
        next_obs = torch.tensor(next_obs).float()
        obs = torch.tensor(obs).float()
        done = torch.tensor(done).float()
        
        with torch.no_grad():
            next_qs = self.q_net_target(next_obs)
            next_q = next_qs.max(dim=1).values # batch対応
            target = reward + (1-done)*self.gamma*next_q
        base_qs = self.q_net(obs)
        base_q = base_qs[np.arange(self.batch_size),action]
        loss = self.criterion(target,base_q)
        loss.backward()
        self.optimizer.step()
    
    def get_action(self,obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            obs = torch.tensor(obs).float().to(self.device)
            with torch.no_grad():
                qs = self.q_net(obs)
            return qs.argmax().item()

# setting
env = gym.make("MountainCar-v0", goal_velocity=0.1)
device = "cpu"
print(device)
gamma = 0.995
epsilon = 1.0
buffer_size = 10000
batch_size = 32
agent = AgentWithDQN(gamma,epsilon,env.action_space.n,env.observation_space.shape[0],device,buffer_size,batch_size)
num_episodes = 50

def visualize_epsilon(num_episodes,start_epsilon=1,end_epsilon=0.001,epsilon_decay_rate=0.9):
    y = []
    ep = start_epsilon
    for episode in range(num_episodes):
        ep = max(end_epsilon,ep*epsilon_decay_rate)
        y.append(ep)
    
    plt.plot(y)
    plt.show()

# 現在の状態における力学的エネルギーを求める。
def get_energy(obs):
    x = obs[0]
    v = obs[1]
    g = 0.0025
    c = 1/(g*np.sin(3*0.5)+0.07*0.07*0.5) # g*sin(3x)+v^2/2 の最大値。v<=0.07の制約あり。
    res = c*(g*np.sin(3*x)+v**2)
    return res

def train(num_episodes,end_epsilon=0.001,epsilon_decay_rate=0.995,sync_interval=20):
    
    for episode in tqdm(range(num_episodes)):
        obs,info = env.reset()
        done = False
        # first = True
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # if first:
            #     print(x)
            #     first = False
            # reward += (abs(x+0.5)) # これ最強。初期位置がx=-0.5なので、初期位置よりも離れることを一番重要視する設計にする。
            reward = get_energy(next_obs)-get_energy(obs)
            agent.update(obs, action, reward, next_obs, done)
            done = terminated or truncated
            obs = next_obs
        agent.epsilon = max(end_epsilon,agent.epsilon*epsilon_decay_rate)
        if episode % sync_interval == 0:
            agent.sync_qnet()

def visualize_agent(num_episodes=10):
    render_env = gym.make("MountainCar-v0", render_mode="human")
    agent.epsilon = 0
    for episode in range(num_episodes):
        obs, info = render_env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated
            obs = next_obs
            if terminated:
                print("Goal!")
    render_env.close()

# test
train(num_episodes=num_episodes)
env.close()
visualize_agent()
