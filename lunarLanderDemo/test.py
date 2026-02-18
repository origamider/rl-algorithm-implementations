import torch
import numpy

a = [1,2,3]
print(torch.tensor(a,dtype=torch.float32).unsqueeze(0))