from collections import defaultdict
import numpy as np

# モンテカルロ法による方策評価を行うagent
class McEvalAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0:0.25,1:0.25,2:0.25,3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.V = defaultdict(lambda:0)
        self.cnts = defaultdict(lambda:0)
        self.memory = []

    def get_action(self,state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions,p=probs)
    
    def add(self,state,action,reward):
        self.memory.append((state,action,reward))

    def reset(self):
        self.memory.clear()
    
    # 方策評価をする。
    # G^i = i回目のエピソードで得られた収益
    # V_pi_n(s) = (G^1+G^2+..+G^n)/n
    # V_n(s) = V_n-1(s) + (G^n - V_n-1)/n
    # s1->s2の時、
    # G[s1] = r(s1) + G[s2]

    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state,action,reward = data
            G = self.gamma*G + reward
            self.cnts[state] += 1
            self.V[state] += (G-self.V[state])/self.cnts[state]