import sys
import os
sys.path.append(os.pardir)
from environments.gridworld import GridWorld
from algorithms.policy_eval import *
from collections import defaultdict
from algorithms.policy_iter import *

# 方策反復法を利用して、最適方策を求めるよ
env = GridWorld()
gamma = 0.9
pi,V = policy_iter(env,gamma)
env.show_map(V,pi)
show_V(V)