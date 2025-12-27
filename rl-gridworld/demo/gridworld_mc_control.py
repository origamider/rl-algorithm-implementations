import sys
import os
sys.path.append(os.pardir)
from environments.gridworld import GridWorld
from algorithms.mc_control import *
from collections import defaultdict

# 概要
# モンテカルロ法を利用して、最適方策を求めるよ。
# 方策制御を行っている。
# ここでは、行動価値関数q(s,a)を更新し、最終的に最適方策piを求めている。
# 状態価値関数V(s)は更新の必要なし。
# Q_n(s,a) = (G^1+G^2+..+G^n)/n


env = GridWorld()
agent = McControlAgent()
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    agent.reset()
    
    while True:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)
        agent.add(state,action,reward)
        if done:
            agent.update()
            break

        state = next_state

env.render_q(agent.Q)