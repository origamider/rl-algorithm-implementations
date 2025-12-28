import os
import sys
sys.path.append(os.pardir)
from algorithms.q_learning import QLearningAgent
from environments.gridworld import GridWorld

# 概要
# Q学習による方策制御

env = GridWorld()
agent = QLearningAgent()
episodes = 10000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)
        agent.update(state,action,reward,next_state,done)
        if done:
            break
        state = next_state
env.render_q(agent.Q)