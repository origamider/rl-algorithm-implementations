import os
import sys
sys.path.append(os.pardir)
from algorithms.sarsa_off_policy import SarsaOffPolicyAgent
from environments.gridworld import GridWorld

# 概要
# 方策ON型のSARSAによる方策制御
# Q'(St,At) = Q(St,At) + α{Rt + γQ(S_(t+1),A_(t+1)) - Q(St,At)}
# 上記式によってQ関数が更新される。ここでSt,At,Rt,S_(t+1),A_(t+1)の5つの変数を必要とすることから、SARSA(サアサ)と呼ばれる。
# ε-greedy法により、探索と活用のバランスをとる。
# SARSAはTD法の一種。
# DP法のようなブートストラップによる価値関数の逐次更新と、MC法のようなサンプリングデータのみで価値関数を更新できる点が強み。

env = GridWorld()
agent = SarsaOffPolicyAgent()
episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state,reward,done = env.step(action)
        agent.update(state,action,reward,done)
        if done:
            agent.update(next_state,None,None,None)
            break
        state = next_state
env.render_q(agent.Q)