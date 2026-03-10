import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import sys
import os
sys.path.append(os.pardir)
from utils.replay_buffer import ReplayBuffer

class Qnet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

class DQNAgent:
    def __init__(self,env,buffer_size=50000,batch_size=64,gamma=0.9,epsilon=1.0):
        self.replay_buffer = ReplayBuffer(buffer_size,batch_size)
        self.q_net = Qnet(env.observation_space.shape[0],32,env.action_space.n)
        self.q_net_target = Qnet(env.observation_space.shape[0],32,env.action_space.n)
        self.action_size = env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters())
    
    def get_action(self,obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.q_net(torch.tensor(obs,dtype=torch.float32).unsqueeze(dim=0))
            return torch.argmax(qs).item()
    
    def update(self,obs,action,reward,next_obs,done):
        self.replay_buffer.add(obs,action,reward,next_obs,done)
        if len(self.replay_buffer) < self.batch_size:
            return
        sample_obs, sample_action, sample_reward, sample_next_obs, sample_done = self.replay_buffer.sample()
        next_qs = self.q_net_target(sample_next_obs)
        next_q = torch.max(next_qs,dim=1).values
        base_q = torch.gather(input=self.q_net(sample_obs),dim=1,index=sample_action.long().unsqueeze(dim=1)).squeeze(dim=1)
        target = sample_reward + self.gamma*next_q*(1-sample_done)
        self.optimizer.zero_grad()
        loss = self.criterion(base_q,target)
        loss.backward()
        self.optimizer.step()
