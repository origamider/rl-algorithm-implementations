import torch 
from model.vae import VAE
from model.mdnrnn import MDNRNN, gmm_loss
import glob
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import os
from utils.checkpoint import load_checkpoint, save_checkpoint
import wandb

CONFIG = {
    # データ
    "data_glob": "rollouts/*/episode_*.npz",
    "vae_artifact": "vae:v0",              # W&B 上の VAE。None にすると vae_weights のファイルを使う
    "z_mode": "sample_per_batch",            # "sample_per_batch"（公式と同じ）または "mu"
    "val_episodes": 30,
    # モデル
    "latent_dim": 32,
    "actions_dim": 3,
    "hidden_dim": 256,
    "gaussians_dim": 5,
    # 学習
    "batch_size": 32,
    "num_epochs": 20,
    "lr": 1e-3,
    "grad_clip_value": 1.0,                  # 公式 rnn.py の tf.clip_by_value と同じ
    "seed": 0,
}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_vae(run, cfg, device):
    artifact = run.use_artifact(cfg["vae_artifact"])   # W&B 上で「この VAE → この MDN-RNN」の依存関係が記録される
    path = os.path.join(artifact.download(), "vae.pt")
    ckpt = load_checkpoint(path, map_location=device)
    vae_cfg = ckpt["config"]
    vae = VAE(img_channels=vae_cfg["img_channels"], latent_dim=vae_cfg["latent_dim"]).to(device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    return vae, path

# それぞれのデータから、VAEを使ってmu,sigma,actionを返す。
# なぜなら、学習せずに推論だけをするため、必要ないからである。
@torch.no_grad()
def encode_episodes(vae, files, device):
    mus, sigmas, actions = [], [], []   # list of (seq_len, latent_dim) / (seq_len, latent_dim) / (seq_len, actions_dim)
    for f in tqdm(files, desc="encode"):
        data = np.load(f)
        obs = data["observations"]  # shape: (seq_len, height, width, img_channels)
        x = torch.from_numpy(obs).permute(0, 3, 1, 2).to(device).float() / 255.0 # shape: (seq_len, img_channels, height, width)
    
        mu, sigma = vae.encoder(x)  # shape: (seq_len, latent_dim)
        mus.append(mu.cpu().numpy())
        sigmas.append(sigma.cpu().numpy())
        actions.append(data["actions"].astype(np.float32))
    return mus, sigmas, actions

# パディング処理。LSTMに流す際、バッチごとにseq_lenが異なると上手く行かない。
# そこで、seq_lenの長さを固定し、余った末尾はmask=0として学習に反映させないようにさせる。
def pad_episodes(mus, sigmas, actions, device):
    # 一番長いエピソードに合わせて0で埋め、本物の部分だけ mask=1 にする
    n_episodes = len(mus)
    max_seq_len = max(len(m) for m in mus)
    latent_dim, actions_dim = mus[0].shape[1], actions[0].shape[1]
    mu_arr = np.zeros((n_episodes, max_seq_len, latent_dim), dtype=np.float32)
    sigma_arr = np.zeros((n_episodes, max_seq_len, latent_dim), dtype=np.float32)
    action_arr = np.zeros((n_episodes, max_seq_len, actions_dim), dtype=np.float32)
    mask = np.zeros((n_episodes, max_seq_len), dtype=np.float32)
    for i, (m, s, a) in enumerate(zip(mus, sigmas, actions)):
        seq_len = len(m)
        mu_arr[i, :seq_len] = m
        sigma_arr[i, :seq_len] = s
        action_arr[i, :seq_len] = a
        mask[i, :seq_len] = 1.0
    # nn.LSTMの入力は(seq_len, batch, dim)のため。
    return {
        "mu": torch.from_numpy(mu_arr).permute(1, 0, 2).to(device),          # shape: (max_seq_len, n_episodes, latent_dim)
        "sigma": torch.from_numpy(sigma_arr).permute(1, 0, 2).to(device),    # shape: (max_seq_len, n_episodes, latent_dim)
        "action": torch.from_numpy(action_arr).permute(1, 0, 2).to(device),  # shape: (max_seq_len, n_episodes, actions_dim)
        "mask": torch.from_numpy(mask).permute(1, 0).to(device),             # shape: (max_seq_len, n_episodes)
    }

def make_batch(data, idx, z_mode):
    mu = data["mu"][:, idx, :]        # shape: (max_seq_len, batch_size, latent_dim)
    sigma = data["sigma"][:, idx, :]  # shape: (max_seq_len, batch_size, latent_dim)
    if z_mode == "sample_per_batch":
        z = mu + sigma * torch.randn_like(mu)   # バッチごとにサンプル（公式 rnn_train.py の random_batch と同じ）
    else:
        z = mu
    z_in = z[:-1, :, :]                 # z_0 ... z_{T-2}  shape: (max_seq_len-1, batch_size, latent_dim)
    a_in = data["action"][:-1, idx, :]  # a_0 ... a_{T-2}  shape: (max_seq_len-1, batch_size, actions_dim)
    z_tar = z[1:, :, :]                 # z_1 ... z_{T-1}  shape: (max_seq_len-1, batch_size, latent_dim)（z はすでに idx で選んであるので、ここでは idx を使わない）
    m_tar = data["mask"][1:, idx]       # 正解 z_{t+1} が本物なら 1  shape: (max_seq_len-1, batch_size)
    return z_in, a_in, z_tar, m_tar

# 検証では、シャッフルせずにバッチで精度を測る
@torch.no_grad()
def evaluate(mdnrnn, data, batch_size, z_mode):
    mdnrnn.eval()
    n = data["mu"].shape[1]
    total_loss, total_steps = 0.0, 0.0
    for i in range(0, n, batch_size):
        idx = torch.arange(i, min(i + batch_size, n), device=data["mu"].device)
        z_in, a_in, z_tar, m_tar = make_batch(data, idx, z_mode)
        pred_mus, pred_sigmas, logpi = mdnrnn(a_in, z_in)
        loss = gmm_loss(pred_mus, pred_sigmas, logpi, z_tar, mask=m_tar)
        n_valid = m_tar.sum().item()
        total_loss += loss.item() * n_valid
        total_steps += n_valid
    mdnrnn.train()
    return total_loss / total_steps

def main():
    with wandb.init(project="world-models-carracing", job_type="train-mdnrnn", config=CONFIG) as run:
        cfg = run.config
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        device = get_device()

        vae, vae_path = load_vae(run, cfg, device)
        
        # エピソード単位でランダムに train / val に分ける
        files = sorted(glob.glob(cfg["data_glob"]))
        order = np.random.default_rng(cfg["seed"]).permutation(len(files))
        val_files = [files[i] for i in order[:cfg["val_episodes"]]]
        train_files = [files[i] for i in order[cfg["val_episodes"]:]]

        train = pad_episodes(*encode_episodes(vae, train_files, device), device)
        val = pad_episodes(*encode_episodes(vae, val_files, device), device)
        
        n_train = len(train_files)
        run.config.update({
            "device": str(device),
            "vae_path": vae_path,
            "n_train_episodes": n_train,
            "n_val_episodes": len(val_files),
            "n_train_steps": int(train["mask"].sum().item()),   # 本物のフレーム数
        })
        mdnrnn = MDNRNN(latent_dim=cfg["latent_dim"], actions_dim=cfg["actions_dim"],
                        hidden_dim=cfg["hidden_dim"], gaussians_dim=cfg["gaussians_dim"]).to(device)
        optimizer = optim.Adam(mdnrnn.parameters(), lr=cfg["lr"])
        
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/loss", step_metric="epoch", summary="min")
        run.watch(mdnrnn, log="gradients", log_freq=50)
        
        os.makedirs("weights", exist_ok=True)
        best_path = f"weights/mdnrnn_{run.id}_best.pt"
        last_path = f"weights/mdnrnn_{run.id}_last.pt"
        best_val = float("inf")
        
        for epoch in tqdm(range(cfg["num_epochs"])):
            perm = torch.randperm(n_train, device=device)
            total_loss, total_steps = 0.0, 0.0
            
            for i in range(0, n_train, cfg["batch_size"]):
                idx = perm[i:i + cfg["batch_size"]]
                z_in, a_in, z_tar, m_tar = make_batch(train, idx, cfg["z_mode"])

                optimizer.zero_grad()
                pred_mus, pred_sigmas, logpi = mdnrnn(a_in, z_in)
                loss = gmm_loss(pred_mus, pred_sigmas, logpi, z_tar, mask=m_tar)
                loss.backward()
                
                # ここではmdnrnnの全パラメータの勾配をまとめた大きさのようなものを求めている。
                grad_norm = torch.norm(torch.stack([p.grad.norm() for p in mdnrnn.parameters() if p.grad is not None]))
                torch.nn.utils.clip_grad_value_(mdnrnn.parameters(), clip_value=cfg["grad_clip_value"])
                optimizer.step()
                
                n_valid = m_tar.sum().item()
                total_loss += loss.item() * n_valid
                total_steps += n_valid
                run.log({"batch/loss": loss.item(), "batch/grad_norm": grad_norm.item()})
            
            train_loss = total_loss / total_steps
            val_loss = evaluate(mdnrnn, val, cfg["batch_size"], cfg["z_mode"])
            run.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss})
            
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, mdnrnn, optimizer, cfg, epoch,
                                {"train_loss": train_loss, "val_loss": val_loss}, run=run)
            
        artifact = wandb.Artifact("mdnrnn", type="model", metadata={**dict(cfg), "best_val_loss": best_val})
        artifact.add_file(best_path, name="mdnrnn.pt")
        run.log_artifact(artifact)

if __name__ == "__main__":
    main()


