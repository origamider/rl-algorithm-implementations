from collections import defaultdict
import numpy as np

class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0:0.25,1:0.25,2:0.25,3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.V = defaultdict(lambda:0)

    def get_action(self,state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions,p=probs)

    # 方策評価する
    # V'(S_t)=V(S_t)+α{R_t+γV(S_t+1)-V(S_t)}
    # モンテカルロ法と違って、ゴールまでVを求める必要がない。
    def eval(self,state,reward,next_state,done):
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V
        self.V[state] += (target-self.V[state])*self.alpha