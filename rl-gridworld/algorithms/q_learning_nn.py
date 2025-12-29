import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# 概要
# ニューラルネットワークを使用したQ学習Agentです。
class QNet(nn.Module):
    def __init__(self,input_size=12,hidden_size=64,output_size=4):
        super().__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        val = self.l1(x)
        val = self.l2(val)
        return val

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4
        self.q_model = QNet()
        self.optimizer = optim.SGD(self.q_model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def get_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.q_model(state)
            return qs.data.argmax()
    
    # Q'(St,At) = Q(St,At) + α{Rt + γmax_a(Q(S_(t+1),a) - Q(St,At))}
    # ここでは、Qの値をNNで求める。
    # 返り値:Q関数の損失値
    def update(self,state,action,reward,next_state,done):
        self.optimizer.zero_grad()
        done = int(done)
        next_qs = self.q_model(next_state)
        next_q = next_qs.max(axis=1)[0]# 返り値が2つある
        target = reward + (1-done)*self.gamma*next_q
        qs = self.q_model(state) # 返り値:[[Q_up,Q_down,Q_left,Q_right]] shape:(1,4)
        q = qs[:,action] # Q(St,At)を取得。バッチ対応でこのような表記になっているが、バッチ=1の場合はqs[0][action]でも可。
        loss = self.criterion(target,q)
        loss.backward()
        self.optimizer.step()
        
        return loss.data