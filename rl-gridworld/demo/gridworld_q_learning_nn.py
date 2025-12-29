import os
import sys
import torch
import numpy as np
sys.path.append(os.pardir)
from algorithms.q_learning_nn import QLearningAgent
from environments.gridworld import GridWorld
import matplotlib.pyplot as plt

# 概要
# ニューラルネットワークを用いたQ学習による方策制御

# (h,w)のgrid上において、状態(i,j)->(1,h*i+j)を1,それ以外を0とする。
def one_hot(state):
    H,W = 3,4
    res = np.zeros(H*W) # shape:(H*W,)
    i,j = state
    res[H*i+j] = 1.0
    res = res.reshape(1,-1)
    res = torch.from_numpy(res).float() # NumPyのデータ型float64からPyTorchのfloat32に変換。
    return res


env = GridWorld()
agent = QLearningAgent()
episodes = 10000
loss_history = []
for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss,ct = 0,0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state,action,reward,next_state,done)
        total_loss += loss
        ct += 1
        state = next_state
    
    average_loss = total_loss / ct
    loss_history.append(average_loss)

Q = {}
for state in env.states():
    for action in range(4):
        q = agent.q_model(one_hot(state))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)