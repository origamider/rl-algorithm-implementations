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
vae.load_state_dict(torch.load("vae_weights.pt", map_location=device))
vae.eval()

files = sorted(glob.glob("rollouts/episode_*.npz"))

episodes_z = [] # list of (seq_len, latent_dim), length = n_episodes
episodes_action = []

with torch.no_grad():
    for f in tqdm(files):
        data = np.load(f)
        obs = data['observations']
        actions = data['actions']
        
        # print(f"obs.shape = {obs.shape}")
        x = torch.tensor(obs).permute(0,3,1,2).to(device)
        mu, sigma = vae.encoder(x)
        z_t = reparameterize(mu, sigma)
        episodes_z.append(z_t.cpu().numpy())
        episodes_action.append(actions)

latent_dim = 32
actions_dim = 3
hidden_dim = 256
gaussians_dim = 5

mdnrnn = MDNRNN(latent_dim=latent_dim, actions_dim=actions_dim, hidden_dim=hidden_dim, gaussians_dim=gaussians_dim)
optimizer = optim.Adam(mdnrnn.parameters(),lr=1e-3)
mdnrnn = mdnrnn.to(device)

zs = np.stack(episodes_z) # (n_episodes, 1000, latent_dim)
actions = np.stack(episodes_action) # (n_episodes, 1000, action_dim)
print(zs.shape)
z_input = zs[:,:-1,:] # (n_episodes, 999, latent_dim)
action_input = actions[:,:-1,:] # (n_episodes, 999, latent_dim)
z_target = zs[:,1:,:] # (n_episodes, 999, latent_dim)

# nn.LSTMの入力は(seq_len, batch, dim)のため。
z_input = torch.tensor(z_input,dtype=torch.float32).permute(1,0,2)
action_input = torch.tensor(action_input,dtype=torch.float32).permute(1,0,2)
z_target = torch.tensor(z_target,dtype=torch.float32).permute(1,0,2)

batch_size = 32
num_episodes = z_input.shape[1]
num_epochs = 20

for epoch in tqdm(range(num_epochs)):
    perm = torch.randperm(num_episodes)
    total_loss = 0.0
    
    for i in range(0, num_episodes, batch_size):
        idx = perm[i:i+batch_size]
        
        z_in = z_input[:,idx,:].to(device)
        a_in = action_input[:,idx,:].to(device)
        z_tar = z_target[:,idx,:].to(device)
        optimizer.zero_grad()
        mus, sigmas, logpi = mdnrnn(a_in, z_in)
        loss = gmm_loss(mus, sigmas, logpi, z_tar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    print(f"epoch {epoch+1}/{num_epochs} loss={total_loss/num_episodes:.4f}")

torch.save(mdnrnn.state_dict(),"mdnrnn_weights.pt")



