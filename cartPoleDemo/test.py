import torch

obs = [1,2,3]
print(torch.max(torch.tensor(obs,dtype=torch.float32),dim=0))