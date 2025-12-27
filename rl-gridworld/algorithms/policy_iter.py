import numpy as np
from collections import defaultdict
import os
import sys
sys.path.append(os.pardir)
from algorithms.policy_eval import *

def argmax(tmp):
    sz = len(tmp)
    maxv = -1
    res = 0
    for i in range(sz):
        if maxv < tmp[i]:
            maxv = tmp[i]
            res = i
    return res

# それぞれの状態sについて最適な移動を求める。(最適方策)
def greedy_policy(V,env,gamma):
    pi = {}
    for state in env.states():
        action_values = {}
        for action in range(4):
            next_state = env.next_state(state,action)
            reward = env.reward(state,action,next_state)
            action_values[action] = reward + gamma*V[next_state]
        max_action = argmax(action_values)
        action_probs = {0:0,1:0,2:0,3:0}
        action_probs[max_action] = 1
        pi[state] = action_probs
    return pi

# 方策が更新されなくなるまで更新する。(最適方策を求める)
def policy_iter(env,gamma):
    pi = defaultdict(lambda : {0:0.25,1:0.25,2:0.25,3:0.25})
    V = defaultdict(lambda : 0)
    while True:
        V = policy_eval(pi,V,env,gamma)
        new_pi = greedy_policy(V,env,gamma)
        if new_pi == pi:
            break
        pi = new_pi

    return pi,V