from collections import deque
import numpy as np
import random

# 経験再生実装
class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    
    # バッファからランダムにバッチ数だけ取得。
    def get_batch(self):
        data = random.sample(self.buffer,self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.stack([x[1] for x in data])
        reward = np.stack([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.stack([x[4] for x in data])
        return state,action,reward,next_state,done


