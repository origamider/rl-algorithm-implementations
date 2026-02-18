import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import time

class ReplayBuffer:
    def __init__(self,batch_size,buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self,obs,action,reward,next_obs,done):
        self.buffer.append((obs,action,reward,next_obs,done))
    
    def sample(self):
        data = random.sample(self.buffer,self.batch_size)
        obs = torch.tensor(np.stack([x[0] for x in data]),dtype=torch.float32) # (batch_size,8)
        action = torch.tensor(np.stack([x[1] for x in data]),dtype=torch.float32) # (batch_size,2)
        reward = torch.tensor(np.stack([x[2] for x in data]),dtype=torch.float32).unsqueeze(dim=1) # (batch_size,)->(batch_size,1)
        next_obs = torch.tensor(np.stack([x[3] for x in data]),dtype=torch.float32)
        done = torch.tensor(np.stack([x[4] for x in data]),dtype=torch.float32).unsqueeze(dim=1) # [1.0,1.0,0.0,1.0]みたいな。(batch_size,)を、(batch_size,1)に変換！
        return obs, action, reward, next_obs, done


class Critic(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3 = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()
    
    def forward(self,obs,action):
        x = torch.cat([obs,action],dim=1) # (batch_size,8)+(batch_size,2) = (batch_size,10)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)

        return x

class Actor(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,output_max,output_min):
        super().__init__()
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3 = nn.Linear(hidden_dim,output_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.action_scale = torch.tensor((output_max-output_min)/2,dtype=torch.float32)
        self.action_bias = torch.tensor((output_max+output_min)/2,dtype=torch.float32)
    
    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.tanh(self.l3(x))
        return self.action_scale*x + self.action_bias

# setting
env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,enable_wind=False, wind_power=15.0, turbulence_power=1.5)
# print(env.observation_space.shape)
action_dim = env.action_space.shape[0] # action_dim=2
obs_dim = env.observation_space.shape[0] # obs_dim=8

actor = Actor(obs_dim,256,action_dim,env.action_space.high,env.action_space.low)
actor_target = Actor(obs_dim,256,action_dim,env.action_space.high,env.action_space.low)
actor_optimizer = optim.Adam(actor.parameters())
critic = Critic(obs_dim+action_dim,256,1)
critic_target = Critic(obs_dim+action_dim,256,1)
critic_optimizer = optim.Adam(critic.parameters())
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(32,10000)

# hyperparameters
num_episodes = 200
exploration_noise = 0.1
learning_starts = 200
total_step = 0
tau = 0.001
gamma = 0.99
# train
start_time = time.time()
print("Training Start!!")
for episode in tqdm(range(num_episodes)):
    obs, info = env.reset()
    done = False
    while not done:
        total_step += 1
        if total_step < learning_starts:
            actions = env.action_space.sample()
        else:
            with torch.no_grad():
                actions = actor(torch.tensor(obs,dtype=torch.float32).unsqueeze(dim=0)) #(2,)->(1,2) イメージ:[1,1]->[[1,1]]
                actions += torch.normal(0,actor.action_scale*exploration_noise)
                actions = actions.cpu().numpy()[0].clip(env.action_space.low,env.action_space.high) #actions.cpu().numpy()は(1,2)だが、[0]を付与することで、(2,)になる。

        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        replay_buffer.add(obs, actions, reward, next_obs, done)
        obs = next_obs

        if total_step > learning_starts:
            sample_obs, sample_action, sample_reward, sample_next_obs, sample_done = replay_buffer.sample()
            target = sample_reward + gamma*critic_target(sample_next_obs,actor_target(sample_next_obs))*(1-sample_done)
            base_q = critic(sample_obs,sample_action)
            
            # update critic
            critic_loss = criterion(target,base_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # update actor
            actor_loss = -critic(sample_obs,actor(sample_obs)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            for param,target_param in zip(critic.parameters(),critic_target.parameters()):
                target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
            for param,target_param in zip(actor.parameters(),actor_target.parameters()):
                target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)

elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"\nTraining finished in {minutes}m {seconds}s")

env.close()
env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,enable_wind=False, wind_power=15.0, turbulence_power=1.5,render_mode="human")

for episode in range(10):
    obs,info = env.reset()
    done = False
    while not done:
        action = actor(torch.tensor(obs,dtype=torch.float32).unsqueeze(0)) #(8,)->(1,8) バッチ化
        action = action.detach().numpy()[0] # (1,8)->(8,)に変換
        next_obs,reward,terminated,truncated,info = env.step(action)
        done = terminated or truncated
        obs = next_obs



