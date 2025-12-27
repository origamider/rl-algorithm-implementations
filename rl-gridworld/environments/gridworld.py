import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.pardir)
import common.gridworld_render as render_helper
class GridWorld:
    def __init__(self):
        self.map = np.array(
            [[0,0,0,1.0],
            [0,None,0,-1],
            [0,0,0,0]]
        )
        self.action_pattern = np.array(["UP","DOWN","LEFT","RIGHT"])
        self.start_state = (0,0)
        self.wall_state = (1,1)
        self.goal_state = (0,3)
        self.agent_state = self.start_state
    
    @property
    def height(self):
        return len(self.map)
    
    @property
    def width(self):
        return len(self.map[0])
    
    def states(self): # yieldを使用すれば、わざわざ配列h*wを作らずに済む
        for i in range(self.height):
            for j in range(self.width):
                yield (i,j)
    
    # 次の状態を返す
    def next_state(self,state,action):
        action_move = [(-1,0),(1,0),(0,-1),(0,1)] # 上下左右
        move = action_move[action]
        next_state = (state[0]+move[0],state[1]+move[1])
        nx,ny = next_state
        
        if nx<0 or nx>=self.height or ny<0 or ny>=self.width:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state
        
        return next_state
    
    # 実際に次の状態に移動する
    def step(self,action):
        state = self.agent_state
        next_state = self.next_state(state,action)
        reward = self.reward(state,action,next_state)
        done = (next_state == self.goal_state)
        self.agent_state = next_state

        return next_state,reward,done

    # r(s,a,s')に対応してる。
    def reward(self,state,action,next_state):
        return self.map[next_state] # numpyの場合、map[1][2]をmap[1, 2]ともかける。ただ、listだったら無理。
    
    # 現在の状態をスタートの時点に戻す。
    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state
    
    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.map, self.goal_state,
                                        self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.map,self.goal_state,self.wall_state)
        renderer.render_q(q, print_value)