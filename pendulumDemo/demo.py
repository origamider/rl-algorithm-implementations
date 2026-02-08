import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import deque
import random

# DDPG(Deep Deterministic Policy Gradient)を使用してPendulumを攻略する。

class ReplayBuffer:
    def __init__(self,buffer_size=10000,batch_size=32):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self,obs,action,reward,next_obs,done):
        self.buffer.append((obs,action,reward,next_obs,done))
    
    def sample(self):
        #reward,doneはスカラー値として返ってくるため、unsqueezeでbatch対応させる。
        data = random.sample(self.buffer,self.batch_size)
        obs = torch.tensor(np.stack([x[0] for x in data]),dtype=torch.float32)
        action = torch.tensor(np.stack([x[1] for x in data]),dtype=torch.float32)
        reward = torch.tensor(np.stack([x[2] for x in data]),dtype=torch.float32).unsqueeze(dim=1)
        next_obs = torch.tensor(np.stack([x[3] for x in data]),dtype=torch.float32)
        done = torch.tensor(np.stack([x[4] for x in data]),dtype=torch.float32).unsqueeze(dim=1)
        
        return obs, action, reward, next_obs, done


# 状態を入力として、 Q値が最大となるaction(連続値)を返す。
class Actor(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.l1 = nn.Linear(env.observation_space.shape[0],256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,env.action_space.shape[0])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.action_scale = torch.tensor((env.action_space.high - env.action_space.low)/2,dtype=torch.float32)
        self.action_bias = torch.tensor((env.action_space.high + env.action_space.low)/2,dtype=torch.float32)
    
    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        x = self.tanh(x) # -1<=return<=1
        return self.action_scale*x + self.action_bias

class Critic(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.l1 = nn.Linear(env.observation_space.shape[0]+env.action_space.shape[0],256) #入力は状態と行動
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,1)
        self.relu = nn.ReLU()
    
    def forward(self,obs,action):
        x = torch.cat([obs,action],dim=1) #obs:(batch_size,3)+action:(batch_size,1)=(batch_size,4)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

env = gym.make("Pendulum-v1")
actor = Actor(env)
actor_target = Actor(env)
actor_target.load_state_dict(actor.state_dict()) #初期の重みを同じにする。
actor_optimizer = optim.Adam(actor.parameters())
critic = Critic(env)
critic_target = Critic(env)
critic_target.load_state_dict(critic.state_dict()) #初期の重みを同じにする。
critic_optimizer = optim.Adam(critic.parameters())
criterion = nn.MSELoss()

replay_buffer = ReplayBuffer()

# hyperparameter
num_episodes = 100
exploration_noise = 0.1
learning_starts = 1000
total_step = 0
gamma = 0.99
tau = 0.005
# train
for episode in tqdm(range(num_episodes)):
    obs, info = env.reset()
    done = False
    while not done:
        total_step += 1
        if total_step < learning_starts:
            actions = env.action_space.sample()
        else:
            with torch.no_grad():
                tmp_obs = torch.tensor(obs,dtype=torch.float32).unsqueeze(dim=0)#(3)->(1,3)
                actions = actor(tmp_obs)
                noise = np.random.normal(0,actor.action_scale.cpu().numpy()*exploration_noise)
                actions = actions.cpu().numpy()[0]
                actions = np.clip(actions+noise,env.action_space.low, env.action_space.high)
        
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        replay_buffer.add(obs, actions, reward, next_obs, done)
        obs = next_obs
        
        if total_step > learning_starts:
            b_obs, b_action, b_reward, b_next_obs, b_done = replay_buffer.sample()
            with torch.no_grad():
                target = b_reward + gamma*critic_target(b_next_obs,actor_target(b_next_obs))*(1-b_done)
            base_q = critic(b_obs,b_action)
            critic_loss = criterion(target,base_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            actor_loss = -critic(b_obs,actor(b_obs)).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            for param,target_param in zip(critic.parameters(),critic_target.parameters()):
                target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
            for param,target_param in zip(actor.parameters(),actor_target.parameters()):
                target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)


env_test = gym.make("Pendulum-v1",render_mode="human")
test_num_episodes = 20

for episode in range(test_num_episodes):
    obs, info = env_test.reset()
    done = False
    
    while not done:
        obs = torch.tensor(obs,dtype=torch.float32).unsqueeze(dim=0)
        actions = actor(obs).detach().numpy()[0] #env.step()はnumpy形式なので。
        next_obs, reward, terminated, truncated, info = env_test.step(actions) 
        done = terminated or truncated
        obs = next_obs
env_test.close()