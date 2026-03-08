import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self,env,gamma=0.9,epsilon=1,alpha=0.1):
        self.Q = defaultdict(lambda : 0)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = env.action_space.n
        self.alpha = alpha
    
    def update(self,obs,action,reward,next_obs,done):
        next_qs = [self.Q[(next_obs,a)] for a in range(self.action_size)]
        target = reward + self.gamma*np.max(next_qs)*(1-done)
        base = self.Q[(obs,action)]
        self.Q[(obs,action)] += self.alpha*(target-base)
    
    def get_action(self,obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[(obs,a)] for a in range(self.action_size)]
            return np.argmax(qs)
