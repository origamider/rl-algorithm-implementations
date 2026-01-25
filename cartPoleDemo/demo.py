import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from tqdm import tqdm

class QNet(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input,n_hidden)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(n_hidden,n_output)
    
    def forward(self,x):
        val = self.l1(x)
        val = self.relu(val)
        val = self.l2(val)
        return val

class AgentWithQLearning:
    def __init__(self,n_action,n_hidden,n_obs,gamma=0.95,start_epsilon=1.0):
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.Q = QNet(n_obs,n_hidden,n_action)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters())
        self.n_action = n_action
    
    def get_action(self,obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_action)
        else:
            obs = torch.tensor(obs).float()
            with torch.no_grad():
                qs = self.Q(obs)
            return qs.argmax().item()
            
    def update(self,obs,action,reward,next_obs,done):
        self.optimizer.zero_grad()
        if done:
            target = reward
        else:
            next_obs = torch.tensor(next_obs).float()
            next_qs = self.Q(next_obs)
            next_q = next_qs.max().item()
            target = reward + self.gamma*next_q
        
        target = torch.tensor(target).float()
        obs = torch.tensor(obs).float()
        base_q = self.Q(obs)[action]
        base_q = base_q.unsqueeze(0)
        loss = self.criterion(target,base_q)
        loss.backward()
        self.optimizer.step()
    
env = gym.make("CartPole-v1", render_mode="rgb_array")
print(env.action_space)
print(env.observation_space.shape)
agent = AgentWithQLearning(env.action_space.n,64,env.observation_space.shape[0])
num_episodes = 1000
end_epsilon = 0.001
epsilon_decay_rate = 0.9995

def train(num_episodes,end_epsilon,epsilon_decay_rate):
    for episode in tqdm(range(num_episodes)):
        obs,info = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(obs,action,reward,next_obs,done)
            obs = next_obs
        agent.epsilon = max(end_epsilon,agent.epsilon*epsilon_decay_rate)
    env.close()

def test(num_episodes=1000):
    agent.epsilon = 0.0
    env = gym.make("CartPole-v1",render_mode="human")
    for episode in tqdm(range(num_episodes)):
        obs,info = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
    env.close()

train(num_episodes,end_epsilon,epsilon_decay_rate)
test()


