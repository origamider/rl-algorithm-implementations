import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class GridWorld:
    def __init__(self):
        self.map = np.array(
            [[0,0,0,1.0],
            [0,None,0,-1],
            [0,0,0,0]]
        )
        self.action_pattern = np.array(["UP","DOWN","LEFT","RIGHT"])
        self.wall_state = (1,1)
        self.goal_state = (0,3)
    
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

    # r(s,a,s')に対応してる。
    def reward(self,state,action,next_state):
        return self.map[next_state] # numpyの場合、map[1][2]をmap[1, 2]ともかける。ただ、listだったら無理。
    
    ## グリッド表示メソッド(AI丸投げ)
    def show_map(self, v_dict=None, pi_dict=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        # 座標系の初期設定（最小限）
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.grid(True, which='minor', color='black')
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)

        for (r, c) in self.states():
            # 1. 値と色の判定（一括処理）
            is_wall = (r, c) == self.wall_state
            val = v_dict.get((r, c), 0) if v_dict else self.map[r, c]
            color = 'gray' if is_wall else ('green' if val > 0 else 'red')
            alpha = 0.3 if is_wall else min(abs(val) / (max(v_dict.values()) if v_dict else 1), 1) * 0.5

            # 2. マスの描画
            ax.add_patch(patches.Rectangle((c-0.5, r-0.5), 1, 1, color=color, alpha=alpha))
            
            # 3. テキストの描画
            if is_wall:
                label = "WALL"
            else:
                label = f"{val:.2f}"
                # 方策がある場合、移動方向を追加
                if pi_dict:
                    action_probs = pi_dict.get((r, c), {})
                    if action_probs:
                        # 確率1.0の行動を探す（決定的方策の場合）
                        best_actions = [action for action, prob in action_probs.items() if prob == 1.0]
                        if best_actions:
                            arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
                            action_arrows = [arrows[action] for action in best_actions]
                            label += f"\n{''.join(action_arrows)}"
            
            ax.text(c, r, label, ha='center', va='center', fontsize=8)
        plt.show()