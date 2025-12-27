import sys
import os
sys.path.append(os.pardir)
from environments.gridworld import GridWorld
from collections import defaultdict
from algorithms.value_iter import *

# 方策反復法を利用して、最適方策を求めるよ
env = GridWorld()
gamma = 0.9
V = defaultdict(lambda:0)
V = value_iter(V,env,gamma)
pi = greedy_policy(V,env,gamma)
env.render_v(V,pi)
show_V(V)