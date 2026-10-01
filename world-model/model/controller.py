import torch
import torch.nn as nn

class Controller(nn.Module):
    """
    Controller
    zs shape:(seq_len, batch_size, latent_dim)
    hs shape:(seq_len, batch_size, hidden_dim)
    """
    def __init__(self, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim+hidden_dim, action_dim)
    
    def forward(self, zs, hs):
        input = torch.cat([zs, hs], dim=-1)
        return self.fc(input)