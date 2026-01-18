import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# 参考URL:https://gymnasium.farama.org/introduction/train_agent/
class BlackJackAgent:
    def __init__(self,env:gym.Env,lr,initial_epsilon,epsilon_decay,last_epsilon,gamma=0.95):
        self.lr = lr
        self.env = env
        self.gamma = gamma
        self.episilon = initial_epsilon
        self.last_episilon = last_epsilon
        self.episilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n)) # type: ignore
        self.training_error = []
    
    # ε-greedy
    def get_action(self,obs):
        if np.random.random() < self.episilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.Q[obs]))
    
    def update(self,obs,action,reward,terminated,next_obs):
        target = reward + self.gamma*((not terminated) * np.max(self.Q[next_obs]))
        temporal_difference = target - self.Q[obs][action]
        self.Q[obs][action] += self.lr*temporal_difference
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        self.episilon = max(self.last_episilon,self.episilon-self.episilon_decay)

# hyperparameters
lr = 0.1
epsilon = 1.0
num_episodes = 400000
epsilon_decay = epsilon / (num_episodes/2)
last_epsilon = 0.1

env = gym.make("Blackjack-v1",sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env,buffer_length=num_episodes)
agent = BlackJackAgent(env=env,lr=lr,initial_epsilon=epsilon,epsilon_decay=epsilon_decay,last_epsilon=last_epsilon)

for episode in tqdm(range(num_episodes)):
    obs,info = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs,action,reward,terminated,next_obs)
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon()

# show result 

# return average of some episode's value
def get_moving_avgs(target,window,convolution_mode):
    return np.convolve(np.array(target).flatten(),np.ones(window)) / window

current_idx = 0
td_errors = []

for length in env.length_queue:
    errors = agent.training_error[current_idx:current_idx+length]
    td_errors.append(np.mean(np.abs(errors)))
    current_idx += length
    

rolling_length = 500
current_idx = 0
training_error_moving_average = get_moving_avgs(td_errors,rolling_length,"valid")

# TD誤差は大体0.6付近で収束するかも
# これはblackjackのゲーム性によるもので、どんなに学習しても最終的に運の要素が残ってしまうからってことかな
plt.figure(figsize=(8,5))
plt.title("TD Error per episode")
plt.plot(range(len(training_error_moving_average)),training_error_moving_average)
plt.ylabel("TD error")
plt.xlabel("episode")
plt.legend()
plt.show()
