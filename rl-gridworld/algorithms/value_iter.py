import numpy as np
from collections import defaultdict
import os
import sys
sys.path.append(os.pardir)
from algorithms.policy_eval import *

# 価値反復法

def argmax(tmp):
    sz = len(tmp)
    maxv = -1
    res = 0
    for i in range(sz):
        if maxv < tmp[i]:
            maxv = tmp[i]
            res = i
    return res

# 価値反復法による価値関数の更新だよ。方策反復法とは違って方策piによる重みがないのが特徴。
# V'(s) = max_a{r(s,a,s') + γV(s')}
def value_iter_onestep(V,env,gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        action_values = []
        for action in range(4):
            next_state = env.step(state,action)
            reward = env.reward(state,action,next_state)
            action_values.append(reward + gamma*V[next_state])

        V[state] = max(action_values)
    return V

# 価値反復法の繰り返し。価値関数が収束するまで実行してるよ
def value_iter(V,env,gamma,threshold=0.001):
    while True:
        oldV = V.copy()
        V = value_iter_onestep(V,env,gamma)
        
        delta = 0
        # 更新された量の最大値を求める。
        for state,value in V.items():
            t = abs(V[state]-oldV[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V

# 最適方策を求める。(方策制御)
# μ'(s) = argmax_a{r(s,a,s') + γV_μ(s')}
def greedy_policy(V,env,gamma):
    pi = {}
    for state in env.states():
        action_values = {}
        for action in range(4):
            next_state = env.step(state,action)
            reward = env.reward(state,action,next_state)
            action_values[action] = reward + gamma*V[next_state]
        max_action = argmax(action_values)
        action_probs = {0:0,1:0,2:0,3:0}
        action_probs[max_action] = 1
        pi[state] = action_probs
    return pi