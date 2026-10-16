import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from model.vae import VAE
from tqdm import tqdm
import wandb
import os
from utils.checkpoint import save_checkpoint

# パラメータの設定
CONFIG = {
    # データ
    "data_glob": "rollouts/20261014_011617/episode_*.npz",
    "val_episodes": 10,            # 最後の10エピソードを検証用にする
    # モデル
    "img_channels": 3,
    "latent_dim": 32,
    "kl_min": 16.0,
    # 学習
    "batch_size": 64,
    "num_epochs": 50,
    "lr": 1e-4,
    "seed": 0,
    # ログ
    "num_log_images": 8,           # W&B に載せる再構成画像の枚数
    "log_images_every": 5,         # 何エポックごとに画像を載せるか
}

# device設定。
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_obs(files):
    return np.concatenate([np.load(f)["observations"] for f in files], axis=0)  # (total_frames, 64, 64, 3)

class RolloutDataset(Dataset):
    def __init__(self, observations):
        # (N, H, W, C) -> (N, C, H, W) に変換しておく（Conv2dの入力形式に合わせる）
        self.observations = torch.from_numpy(observations).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx]

# 評価。
@torch.no_grad()
def evaluate(vae, loader, device, kl_min):
    vae.eval()
    sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    for x in loader:
        x = x.to(device).float() / 255.0
        loss, recon, kl = vae.get_loss(x, kl_min=kl_min, return_parts=True)
        for k, v in zip(sums, (loss, recon, kl)):
            sums[k] += v.item() * len(x)
        n += len(x)
    vae.train()
    return {k: v / n for k, v in sums.items()}

@torch.no_grad()
def reconstruction_images(vae, x):
    # 左に元画像、右に復元画像を並べた画像のリスト。復元はサンプルせず mu から行う
    vae.eval()
    mu, _ = vae.encoder(x)
    x_hat = vae.decoder(mu)
    vae.train()
    pairs = torch.cat([x, x_hat], dim=3)  # (num_images, img_channels, height, 2*width)
    return [wandb.Image(p.permute(1, 2, 0).cpu().numpy()) for p in pairs]

def main():
    with wandb.init(project="world-models-carracing", job_type="train-vae", config=CONFIG) as run:
        cfg = run.config
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        device = get_device()
        files = sorted(glob.glob(cfg["data_glob"]))
        train_files = files[:-cfg["val_episodes"]]
        val_files = files[-cfg["val_episodes"]:]
        train_obs = load_obs(train_files)
        val_obs = load_obs(val_files)
        
        run.config.update({
            "device": str(device),
            "n_train_episodes": len(train_files),
            "n_train_frames": len(train_obs),
            "n_val_frames": len(val_obs),
        })
        train_dataset = RolloutDataset(train_obs)
        val_dataset = RolloutDataset(val_obs)
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256)

        image_idx = np.linspace(0, len(val_dataset) - 1, cfg["num_log_images"]).astype(int)
        fixed_x = val_dataset.observations[image_idx].to(device).float() / 255.0
        
        vae = VAE(img_channels=cfg["img_channels"], latent_dim=cfg["latent_dim"]).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=cfg["lr"])
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("val/loss", step_metric="epoch", summary="min")
        os.makedirs("weights", exist_ok=True)
        best_path = f"weights/vae_{run.id}_best.pt"
        last_path = f"weights/vae_{run.id}_last.pt"
        best_val = float("inf")
        
        for epoch in tqdm(range(cfg["num_epochs"])):
            sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
            for x in train_loader:
                x = x.to(device).float() / 255.0 # obsはuintで保存したため、float変換&[0,1]変換。
                optimizer.zero_grad()
                loss, recon, kl = vae.get_loss(x, kl_min=cfg["kl_min"], return_parts=True)
                loss.backward()
                optimizer.step()
                for k, v in zip(sums, (loss, recon, kl)):
                    sums[k] += v.item() * len(x)
            train = {k: v / len(train_dataset) for k, v in sums.items()}
            val = evaluate(vae, val_loader, device, cfg["kl_min"])

            log = {"epoch": epoch}
            log.update({f"train/{k}": v for k, v in train.items()})
            log.update({f"val/{k}": v for k, v in val.items()})
            if epoch % cfg["log_images_every"] == 0 or epoch == cfg["num_epochs"] - 1:
                log["reconstruction"] = reconstruction_images(vae, fixed_x)
            run.log(log)

            if val["loss"] < best_val:
                best_val = val["loss"]
                save_checkpoint(best_path, vae, optimizer, cfg, epoch, {"train": train, "val": val}, run=run)
        
        save_checkpoint(last_path, vae, optimizer, cfg, epoch, {"train": train, "val": val}, run=run)

        artifact = wandb.Artifact("vae", type="model", metadata={**dict(cfg), "best_val_loss": best_val})
        artifact.add_file(best_path, name="vae.pt")
        run.log_artifact(artifact)   # "latest" は自動で付く

if __name__ == "__main__":
    main()