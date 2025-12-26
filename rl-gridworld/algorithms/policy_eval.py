import os
import sys
sys.path.append(os.pardir)
from environments.gridworld import GridWorld
# 反復方策評価アルゴリズム
# 方策反復法（Policy Iteration）

# 全ての状態における価値関数を求める。
# 決定論的な状態遷移とする。
# V_(k+1)(s) = sigma_a(pi(a|s){r(s,a,s') + γV_k(s')})
def eval_onestep(pi,V,env,gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state] # 上下左右それぞれのactionの確率
        newV = 0
        
        for action,action_prob in action_probs.items():
            next_state = env.step(state,action)
            newV += action_prob*(env.reward(state,action,next_state)+gamma*V[next_state])
        V[state] = newV
        
    return V

# 方策評価
def policy_eval(pi,V,env,gamma,threshold=0.001):
    while True:
        oldV = V.copy()
        V = eval_onestep(pi,V,env,gamma)
        
        delta = 0
        # 更新された量の最大値を求める。
        for state,value in V.items():
            t = abs(V[state]-oldV[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V

def show_V(V):
    for state,value in V.items():
        print(f"{state}: {value}")