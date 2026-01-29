import numpy
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import copy
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt

class Qnet(nn.Module):
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

# 経験再生
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
        obs = np.stack([x[0] for x in data])
        action = np.stack([x[1] for x in data])
        reward = np.stack([x[2] for x in data])
        next_obs = np.stack([x[3] for x in data])
        done = np.stack([x[4] for x in data])
        return obs, action, reward, next_obs, done

class AgentWithDQN:
    def __init__(self,action_size,obs_size,batch_size=64,buffer_size=10000,gamma=0.99,first_episilon=1.0):
        self.gamma = gamma
        self.epsilon = first_episilon
        self.action_size = action_size
        self.obs_size = obs_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size,batch_size)
        self.qnet = Qnet(self.obs_size,128,self.action_size)
        self.qnet_target = Qnet(self.obs_size,128,self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters())
        self.criterion = nn.MSELoss()
    
    def get_action(self,obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            obs = torch.tensor(obs).float()
            with torch.no_grad():
                qs = self.qnet(obs)
            return qs.argmax().item()
            
    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)
        
    def update(self,obs,action,reward,next_obs,done):
        self.optimizer.zero_grad()
        
        self.replay_buffer.add(obs,action,reward,next_obs,done)
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        obs,action,reward,next_obs,done = self.replay_buffer.sample()
        obs = torch.tensor(obs).float()
        reward = torch.tensor(reward).float()
        next_obs = torch.tensor(next_obs).float()
        done = torch.tensor(done).float()
        
        next_q = self.qnet_target(next_obs).max(dim=1).values
        base_qs = self.qnet(obs)
        base_q = base_qs[np.arange(self.batch_size),action]
        target = reward + self.gamma*next_q*(1-done)
        
        loss = self.criterion(target,base_q)
        loss.backward()
        self.optimizer.step()

env = gym.make("Acrobot-v1",render_mode="rgb_array")

agent = AgentWithDQN(int(env.action_space.n),int(env.observation_space.shape[0])) # pyright: ignore[reportOptionalSubscript]

def get_energy(obs):
    cos1, sin1, cos2, sin2, v1, v2 = obs
    # 速さも考慮
    # height = -cos1-(cos1*cos2-sin1*sin2)
    # v = 0.5*(v1**2+v2**2)
    # c = 1/(2 + 0.5*(12.567*12.567+28.274*28.274))
    # return c*(height+v)
    
    # 高さ重視
    height = -cos1-(cos1*cos2-sin1*sin2)
    c = 1/2
    return c*height

def train(num_episodes,end_epsilon=0.001,epsilon_decay_rate=0.99,sync_interval=10):
    reward_history = []
    for episode in tqdm(range(num_episodes)):
        total_reward = 0
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward = (get_energy(next_obs)-get_energy(obs))
            total_reward += reward
            # print(f"reward = {reward}")
            agent.update(obs,action,reward,next_obs,done)
            done = truncated or terminated
            obs = next_obs
        agent.epsilon = max(end_epsilon,agent.epsilon*epsilon_decay_rate)
        if episode % sync_interval == 0:
            agent.sync_qnet()
        reward_history.append(total_reward)
    plt.plot(reward_history)
    plt.title(f'total reward')
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.show()
    
def test(num_episodes=10):
    render_env = gym.make("Acrobot-v1",render_mode="human")
    agent.epsilon = 0.0
    for episode in tqdm(range(num_episodes)):
        obs, info = render_env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = render_env.step(action)
            done = terminated
            obs = next_obs
    render_env.close()

# setting
num_episodes = 30

train(num_episodes)
test()