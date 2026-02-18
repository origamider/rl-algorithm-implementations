import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,output_max,output_min):
        super().__init__()
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3 = nn.Linear(hidden_dim,output_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.action_scale = torch.tensor((output_max-output_min)/2,dtype=torch.float32)
        self.action_bias = torch.tensor((output_max+output_min)/2,dtype=torch.float32)
    
    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.tanh(self.l3(x))
        return self.action_scale*x + self.action_bias

def test_actor():
    print("Starting verification of Actor implementation...")
    
    # Test parameters
    input_dim = 8
    hidden_dim = 256
    output_dim = 2
    # In LunarLander, action range is [-1, 1]
    output_max = 1.0
    output_min = -1.0
    
    print(f"Parameters: output_max={output_max}, output_min={output_min}")
    
    try:
        actor = Actor(input_dim, hidden_dim, output_dim, output_max, output_min)
        
        print(f"Calculated action_scale: {actor.action_scale}")
        print(f"Calculated action_bias: {actor.action_bias}")
        
        # Verify scale is positive
        if actor.action_scale > 0:
            print("SUCCESS: action_scale is positive.")
        else:
            print("FAILURE: action_scale is non-positive!")
            return False

        # Verify torch.normal works with this scale
        exploration_noise = 0.1
        std = actor.action_scale * exploration_noise
        print(f"Testing torch.normal with std={std}")
        
        noise = torch.normal(0, std)
        print(f"Generated noise sample: {noise}")
        print("SUCCESS: torch.normal execution completed without error.")
        return True
        
    except Exception as e:
        print(f"FAILURE: An error occurred during verification: {e}")
        return False

if __name__ == "__main__":
    if test_actor():
        print("\nVERIFICATION PASSED")
    else:
        print("\nVERIFICATION FAILED")
        exit(1)
