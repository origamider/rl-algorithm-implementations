import numpy
from collections import deque
import random
import torch
import numpy as np

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
        action = torch.tensor(np.stack([x[1] for x in data]),dtype=torch.float32)
        reward = torch.tensor(np.stack([x[2] for x in data]),dtype=torch.float32)
        next_obs = torch.tensor(np.stack([x[3] for x in data]),dtype=torch.float32)
        done = torch.tensor(np.stack([x[4] for x in data]),dtype=torch.float32)
        return obs, action, reward, next_obs, done
