import sys
import os
sys.path.append(os.pardir)
from environments.gridworld import GridWorld
from algorithms.td_eval import *
from collections import defaultdict

# 概要
# TD法によって価値関数を求める。


env = GridWorld()
agent = TdAgent()
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    
    while True:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)
        agent.eval(state,reward,next_state,done)
        if done:
            break

        state = next_state

env.render_v(agent.V)