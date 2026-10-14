import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from model.vae import VAE
from tqdm import tqdm

files = sorted(glob.glob("rollouts/20261014_011617/episode_*.npz"))
train_files = files[:190]
val_files = files[190:] # 検証用
all_obs = []
for f in train_files:
    data = np.load(f)
    all_obs.append(data["observations"])  # (T, 64, 64, 3)

all_obs = np.concatenate(all_obs, axis=0)  # (total_frames, 64, 64, 3)

class RolloutDataset(Dataset):
    def __init__(self, observations):
        # (N, H, W, C) -> (N, C, H, W) に変換しておく（Conv2dの入力形式に合わせる）
        self.observations = torch.from_numpy(observations).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx]

train_dataset = RolloutDataset(all_obs)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vae = VAE(img_channels=3, latent_dim=32).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-4)

num_epochs = 50

for epoch in tqdm(range(num_epochs)):
    total_loss = 0.0
    for batch_obs in train_loader:
        batch_obs = batch_obs.to(device).float() / 255.0 # obsはuintで保存したため、float変換&[0,1]変換。
        optimizer.zero_grad()
        loss = vae.get_loss(batch_obs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_obs)
    
    avg_loss = total_loss / len(train_dataset)
    print(f"epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}")

torch.save(vae.state_dict(), "vae_weights.pt")