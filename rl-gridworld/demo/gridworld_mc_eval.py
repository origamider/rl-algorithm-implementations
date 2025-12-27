import sys
import os
sys.path.append(os.pardir)
from environments.gridworld import GridWorld
from algorithms.mc_control import *
from algorithms.mc_eval import *
from collections import defaultdict

# モンテカルロ法を利用して、ランダム方策における価値関数を可視化するよ。
env = GridWorld()
agent = McEvalAgent()
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    agent.reset()
    
    while True:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)
        agent.add(next_state,action,reward)
        if done:
            agent.eval()
            break
        state = next_state

env.render_v(agent.V)