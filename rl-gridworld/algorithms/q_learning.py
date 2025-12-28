from collections import defaultdict
import numpy as np

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        random_probs = {0:0.25,1:0.25,2:0.25,3:0.25}
        self.b = defaultdict(lambda:random_probs) # 挙動方策
        self.pi = defaultdict(lambda:random_probs)
        self.Q = defaultdict(lambda:0)

    def get_action(self,state):
        action_probs = self.b[state] # 挙動方策から取得
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions,p=probs)
    
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
    
    def update(self,state,action,reward,next_state,done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state,a] for a in range(self.action_size)]
            next_q_max = max(next_qs)
        target = reward + self.gamma*next_q_max
        idx = (state,action)
        self.Q[idx] += self.alpha*(target - self.Q[idx])
        self.pi[state] = self.greedy_probs(state,epsilon=0,action_size=self.action_size)
        self.b[state] = self.greedy_probs(state,epsilon=self.epsilon,action_size=self.action_size)