import torch 
import torch.nn as nn
from model.vae import VAE, reparameterize
from model.mdnrnn import MDNRNN, gmm_loss
import glob
import numpy as np
from tqdm import tqdm
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vae = VAE(img_channels=3, latent_dim=32).to(device)
vae.load_state_dict(torch.load("weights/vae_weights_4.pt", map_location=device))
vae.eval()

files = sorted(glob.glob("rollouts/episode_*.npz"))

episodes_mu = [] # list of (seq_len, latent_dim), length = n_episodes
episodes_sigma = []
episodes_action = [] # list of (seq_len, action_dim)

with torch.no_grad():
    for f in tqdm(files):
        data = np.load(f)
        obs = data['observations']
        actions = data['actions']
        x = torch.tensor(obs).permute(0,3,1,2).to(device).float()
        mu, sigma = vae.encoder(x)
        episodes_mu.append(mu.cpu().copy())
        episodes_sigma.append(sigma.cpu().copy())
        episodes_action.append(actions)

n_episodes = len(episodes_mu)
T = max(len(z) for z in episodes_mu) # max seq_len

latent_dim = 32
actions_dim = 3
hidden_dim = 256
gaussians_dim = 5

mdnrnn = MDNRNN(latent_dim=latent_dim, actions_dim=actions_dim, hidden_dim=hidden_dim, gaussians_dim=gaussians_dim)
optimizer = optim.Adam(mdnrnn.parameters(),lr=1e-3)
mdnrnn = mdnrnn.to(device)

mus = np.zeros((n_episodes, T, latent_dim), dtype=np.float32)
sigmas = np.zeros((n_episodes, T, latent_dim), dtype=np.float32)
actions = np.zeros((n_episodes, T, actions_dim), dtype=np.float32)
mask = np.zeros((n_episodes, T), dtype=np.float32)

# Tまでで不足している部分は0として補う。
for i, (m, s, a) in enumerate(zip(episodes_mu, episodes_sigma, episodes_action)):
    seq_len = len(m)
    mus[i, :seq_len] = m
    sigmas[i, :seq_len] = s
    actions[i, :seq_len] = a
    mask[i, :seq_len] = 1.0

# nn.LSTMの入力は(seq_len, batch, dim)のため。
mus = torch.tensor(mus,dtype=torch.float32).permute(1,0,2).to(device)
sigmas = torch.tensor(sigmas,dtype=torch.float32).permute(1,0,2).to(device)
actions= torch.tensor(actions,dtype=torch.float32).permute(1,0,2).to(device)
mask = torch.tensor(mask,dtype=torch.float32).permute(1,0,2).to(device)

batch_size = 32
num_epochs = 20

for epoch in tqdm(range(num_epochs)):
    perm = torch.randperm(n_episodes, device=device)
    total_loss = 0.0
    total_steps = 0.0
    
    for i in range(0, n_episodes, batch_size):
        idx = perm[i:i+batch_size]
        
        mu = mus[:, idx, :]
        sigma = sigmas[:, idx, :]
        z = mu + sigma * torch.randn_like(mu) # (seq_len, batch, dim)
        z_in = z[:-1,:,:]
        a_in = actions[:-1,idx,:]
        z_tar = z[1:,idx,:]
        m_tar = mask[1:, idx]
        optimizer.zero_grad()
        mus, sigmas, logpi = mdnrnn(a_in, z_in)
        loss = gmm_loss(mus, sigmas, logpi, z_tar,mask=m_tar)
        loss.backward()
        torch.nn.utils.clip_grad_value_(mdnrnn.parameters(), clip_value=1.0)
        optimizer.step()
        total_loss += loss.item() * m_tar.sum().item()
        total_steps = m_tar.sum().item()
    print(f"epoch {epoch+1}/{num_epochs} loss={total_loss/total_steps:.4f}")

torch.save(mdnrnn.state_dict(),"weights/mdnrnn_weights.pt")



