import torch
import torch.nn as nn

class Controller(nn.Module):
    """
    Controller
    zs shape:(batch_size, latent_dim)
    hs shape:(batch_size, hidden_dim)
    output shape: (batch_size, action_dim)  [steer, gas, brake]
    
    行動の範囲は論文・公式実装(model.py の get_action)と同じ:
    steer: tanh -> [-1, 1]
    gas  : (tanh + 1) / 2 -> [0, 1]
    brake: clip(tanh, 0, 1) -> [0, 1]
    
    input_mode:
    "z_and_h": a_t = W [z_t, h_t] + b（論文のフルの World Model）
    "z_only"  : a_t = W z_t + b（論文の「V model only」。比較用）

    """
    def __init__(self, latent_dim, hidden_dim, action_dim, input_mode="z_and_h"):
        super().__init__()
        self.input_mode = input_mode
        input_dim = latent_dim + hidden_dim if input_mode == "z_and_h" else latent_dim
        self.fc = nn.Linear(input_dim, action_dim)
    
    def forward(self, zs, hs):
        # input shape: (batch_size, input_dim)
        # 基本的にはbatch_size=1とする。
        input = torch.cat([zs, hs], dim=-1) if self.input_mode == "z_and_h" else zs
        a = torch.tanh(self.fc(input))
        steer = a[:, 0] # shape: (batch_size,)
        gas = (a[:, 1] + 1.0) / 2.0 # shape: (batch_size,)
        brake = a[:, 2].clamp(min=0.0) # shape: (batch_size,)
        
        return torch.stack([steer, gas, brake], dim=-1)