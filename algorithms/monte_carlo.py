from collections import defaultdict
import numpy as np

class MonteCarloAgent:
    def __init__(self,env,alpha=0.1,gamma=0.9,epsilon=1.0):
        self.Q = defaultdict(lambda:0)
        self.alpha = alpha
        self.action_size = env.action_space.n
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = []
    
    def add(self,obs,action,reward):
        self.memory.append((obs,action,reward))
    
    def get_action(self,obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # max_v = -100000
            # res = -1
            # for i in range(self.action_size):
            #     if max_v < self.Q[(obs,i)]:
            #         res = i
            #         max_v = self.Q[(obs,i)]
            qs = [self.Q[(obs,i)] for i in range(self.action_size)]
            return np.argmax(qs)
    
    def reset(self):
        self.memory.clear()
    
    def update(self):
        G = 0
        for data in reversed(self.memory):
            obs, action, reward = data
            G = reward + self.gamma*G
            self.Q[(obs,action)] += (G - self.Q[(obs,action)])*self.alpha