import torch
import torch.nn as nn
from model.vae import VAE
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
import gymnasium as gym

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vae = VAE(img_channels=3, latent_dim=32)
vae.load_state_dict(torch.load("weights/vae_weights_3.pt",map_location=device))

files = glob.glob("rollouts/20261014_011617/episode_*.npz")
test_data = np.load(files[199])
print(test_data['observations'].shape)

# original画像と復元画像の比較。
num_test = 10

fig, axes = plt.subplots(num_test,2,figsize=(10,10))

for i in range(num_test):
    id = random.randint(0, len(test_data['observations']) - 1)
    original = test_data['observations'][id,:,:,:]
    x = torch.tensor(original).permute(2,0,1).unsqueeze(dim=0).float() / 255.0
    x_hat = vae(x)
    x_hat = x_hat.squeeze(dim=0).permute(1,2,0).to('cpu').detach().numpy().copy()
    axes[i][0].imshow(original)
    axes[i][0].set_title("Original")
    axes[i][1].imshow(x_hat)
    axes[i][1].set_title("image restored by latent z")
plt.show()


# VAEのmu,sigma調査

id = random.randint(1,200)
target = test_data['observations'][id,:,:,:]
target = torch.tensor(target,dtype=torch.float32).permute(2,0,1).unsqueeze(dim=0) / 255.0 # uintのため、255で割る。
print(f"target.shape = {target.shape}")
mu, sigma = vae.encoder(target)
print(f"mu.shape = {mu.shape}")
print(f"sigma.shape = {sigma.shape}")
print(f"mu = {mu}")
print(f"sigma = {sigma}")
print(f"mu mean: {torch.mean(mu)}")
print(f"sigma mean: {torch.mean(sigma)}")
