import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import copy
sys.path.append(os.pardir)
from algorithms.replay_buffer import ReplayBuffer

class QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
    def forward(self,x):
        val = self.l1(x)
        val = self.relu(val)
        val = self.l2(val)
        return val

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.001
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2
        self.replay_buffer = ReplayBuffer(self.buffer_size,self.batch_size)
        self.qnet = QNet(4,128,2)
        self.qnet_target = QNet(4,128,2)
        self.optimizer = optim.Adam(self.qnet.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()
    
    # qnet_targetを更新。
    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)
    
    def get_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state_tensor = torch.from_numpy(state).float()
            qs = self.qnet(state_tensor)
            return qs.data.argmax().item()
    
    def update(self,state,action,reward,next_state,done):
        self.optimizer.zero_grad()
        self.replay_buffer.add(state,action,reward,next_state,done)
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        state,action,reward,next_state,done = self.replay_buffer.get_batch()
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).long()
        reward = torch.from_numpy(reward).float()
        next_state = torch.from_numpy(next_state).float()
        done = torch.from_numpy(done).float()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size),action] # actionは0or1。それぞれactionに沿ったq値を取得する。
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(dim=1)[0]
        target = (1-done)*self.gamma*next_q + reward

        loss = self.criterion(q,target)
        loss.backward()
        self.optimizer.step()