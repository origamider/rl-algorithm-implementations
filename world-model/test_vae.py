import torch
import torch.nn as nn
from model.vae import VAE
import numpy as np
import glob
import random
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vae = VAE(img_channels=3, latent_dim=32)
vae.load_state_dict(torch.load("vae_weights.pt",map_location=device))

files = glob.glob("rollouts/episode_*.npz")
test_data = np.load(files[0])
print(test_data['observations'].shape)

id = random.randint(1,200)
original = test_data['observations'][id,:,:,:]
x = torch.tensor(original).permute(2,0,1).unsqueeze(dim=0)

x_hat = vae(x)
x_hat = x_hat.squeeze(dim=0).permute(1,2,0).to('cpu').detach().numpy().copy()

num_test = 10

fig, axes = plt.subplots(num_test,2,figsize=(10,10))

for i in range(num_test):
    id = random.randint(1,200)
    original = test_data['observations'][id,:,:,:]
    x = torch.tensor(original).permute(2,0,1).unsqueeze(dim=0)
    x_hat = vae(x)
    x_hat = x_hat.squeeze(dim=0).permute(1,2,0).to('cpu').detach().numpy().copy()
    axes[i][0].imshow(original)
    axes[i][0].set_title("Original")
    axes[i][1].imshow(x_hat)
    axes[i][1].set_title("image restored by latent z")
plt.show()

