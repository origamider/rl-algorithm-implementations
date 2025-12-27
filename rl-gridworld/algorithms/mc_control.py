from collections import defaultdict
import numpy as np

# 方策制御用agentです。
# Qの更新を、固定値αによる指数移動平均にする。
# ε-greedy法によって、活用+探索を実現

class McControlAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        self.alpha = 0.1
        self.epsilon = 0.1

        random_actions = {0:0.25,1:0.25,2:0.25,3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.Q = defaultdict(lambda:0)
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

    # その状態における、最適方策を求める。
    # Q関数の最も大きいactionを(1-ε)+ε/4,それ以外をε/4とする。(合計1)
    def greedy_probs(self,state,epsilon=0.0,action_size=4):
        qs = []
        for action in range(action_size):
            qs.append(self.Q[(state,action)])
        max_action = int(np.argmax(qs))
        base_prob = epsilon/action_size
        res ={action:base_prob for action in range(action_size)}
        res[max_action] += (1-epsilon)
        return res
    
    # 方策制御
    # 全ての状態における最適方策を更新する。
    # G^i = i回目のエピソードで得られた収益
    # Q_n(s,a) = (G^1+G^2+..+G^n)/n
    # Q_n(s,a) = Q_n-1(s,a) + (G^n - Q_n-1(s,q))/n
    def update(self):
        G = 0
        for data in reversed(self.memory):
            state,action,reward = data
            G = self.gamma*G + reward
            idx = (state,action)
            self.cnts[idx] += 1
            self.Q[idx] += (G-self.Q[idx])*self.alpha

            self.pi[state] = self.greedy_probs(state,epsilon=self.epsilon)