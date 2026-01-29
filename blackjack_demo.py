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
lr = 0.001
epsilon = 1.0
num_episodes = 1000000
epsilon_decay = epsilon / (num_episodes/2)
last_epsilon = 0.01

env = gym.make("Blackjack-v1",sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env,buffer_length=num_episodes)
agent = BlackJackAgent(env=env,lr=lr,initial_epsilon=epsilon,epsilon_decay=epsilon_decay,last_epsilon=last_epsilon)

for episode in tqdm(range(num_episodes)):
    obs,info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs,action,reward,terminated,next_obs)
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon()

def test_agent(num_episodes=1000):
    agent.episilon = 0
    total_rewards = []
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)
    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)
    print(f"win_rate = {win_rate}")
    print(f"average_reward = {average_reward}")

# show result 

# return average of some episode's value
def get_moving_avgs(target,window,convolution_mode):
    return np.convolve(np.array(target).flatten(),np.ones(window)) / window

# evaluate agent
test_agent()

rolling_length = 500

fig,axes = plt.subplots(3,figsize=(12,6))
axes[0].set_title("Rewards per episode")
reward_moving_average = get_moving_avgs(agent.env.return_queue,rolling_length,"valid")
axes[0].plot(range(len(reward_moving_average)),reward_moving_average)
axes[0].set_xlabel("episode")
axes[0].set_ylabel("Rewards")

axes[1].set_title("Length per episode")
length_moving_average = get_moving_avgs(agent.env.length_queue,rolling_length,"valid")
axes[1].plot(range(len(length_moving_average)),length_moving_average)
axes[1].set_xlabel("episode")
axes[1].set_ylabel("Length")


# TD error
axes[2].set_title("TD error per episode")
training_error_moving_average = get_moving_avgs(agent.training_error,rolling_length,"valid")
axes[2].plot(range(len(training_error_moving_average)),training_error_moving_average)
axes[2].set_xlabel("episode")
axes[2].set_ylabel("TD error")



plt.show()
