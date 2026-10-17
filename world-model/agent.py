import numpy as np
import torch
from PIL import Image
from model.vae import VAE
from model.mdnrnn import MDNRNN
from model.controller import Controller
from utils.checkpoint import load_checkpoint

def preprocess(obs):
    # collect.py と同じ前処理: (96, 96, 3) uint8 -> 64x64 にリサイズ -> [0, 1]
    img = Image.fromarray(obs).resize((64, 64), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.array(img, dtype=np.uint8))  # shape: (height, width, img_channels)
    return x.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # shape: (1, img_channels, height, width)

def load_vae(path):
    ckpt = load_checkpoint(path)
    cfg = ckpt["config"]
    vae = VAE(img_channels=cfg["img_channels"], latent_dim=cfg["latent_dim"])
    vae.load_state_dict(ckpt["model_state_dict"])
    return vae.eval()

def load_mdnrnn(path):
    ckpt = load_checkpoint(path)
    cfg = ckpt["config"]
    mdnrnn = MDNRNN(latent_dim=cfg["latent_dim"], actions_dim=cfg["actions_dim"],
                    hidden_dim=cfg["hidden_dim"], gaussians_dim=cfg["gaussians_dim"])
    mdnrnn.load_state_dict(ckpt["model_state_dict"])
    return mdnrnn.eval()

class WorldModelAgent:
    """V (VAE) + M (MDN-RNN) + C (Controller) をまとめたエージェント。1フレームずつ行動を返す。"""

    def __init__(self, vae, mdnrnn, controller, z_mode="sample"):
        self.vae = vae
        self.mdnrnn = mdnrnn
        self.controller = controller
        self.z_mode = z_mode   # "sample": z = mu + sigma * eps（MDN-RNN の学習と同じ）/ "mu": z = mu
        self.reset()

    def reset(self):
        self.hidden = None                                       # LSTM の (h, c)。None は 0 から始めるという意味
        self.h = torch.zeros(1, self.mdnrnn.hidden_dim)          # shape: (1, hidden_dim)

    @torch.no_grad()
    def act(self, obs):
        x = preprocess(obs)                                      # shape: (1, img_channels, height, width)
        mu, sigma = self.vae.encoder(x)                          # shape: (1, latent_dim)
        z = mu + sigma * torch.randn_like(mu) if self.z_mode == "sample" else mu
        action = self.controller(z, self.h)                      # shape: (1, actions_dim)
        self.hidden = self.mdnrnn.step(action, z, self.hidden)   # 次のフレームのために h を更新
        self.h = self.hidden[0][0]                               # (1, 1, hidden_dim) -> (1, hidden_dim)
        return action[0].numpy()                                 # shape: (actions_dim,)

def rollout(agent, env, seed, record=False):
    """1レース走らせて、合計報酬を返す。record=True なら、観測画像のリストも返す。"""
    obs, _ = env.reset(seed=seed)
    agent.reset()
    total_reward = 0.0
    frames = [obs] if record else None
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if record:
            frames.append(obs)
    return (total_reward, frames) if record else total_reward

