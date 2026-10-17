import os
import multiprocessing as mp
import numpy as np
import torch
import gymnasium as gym
from PIL import Image
import wandb
from agent import WorldModelAgent, load_vae, load_mdnrnn, preprocess
from model.controller import Controller
from utils.checkpoint import load_checkpoint
from train_controller import init_worker, evaluate_params

CONFIG = {
    "controller_artifact": "controller:latest",
    "n_episodes": 100,          # 論文と同じく100コースの平均で評価する
    "n_videos": 3,              # 映像を記録するコース数
    "num_workers": 10,
    "seed": 12345,              # 学習で使っていないコースにするため、学習とは別の seed
    "video_frame_skip": 2,      # 映像は2フレームに1枚（そのままだと1本で数百MBになるため）
}

def to_video(frames, size):
    # list of (height, width, 3) uint8 -> (time, 3, size_h, size_w) uint8（wandb.Video の形式）
    # PIL の大きさは (width, height)らしい。sizeには、 (width, height)に合わせるように。
    resized = [np.array(Image.fromarray(f).resize(size, Image.Resampling.NEAREST)) for f in frames]
    return np.stack(resized).transpose(0, 3, 1, 2)

@torch.no_grad()
def record_episode(agent, vae, seed):
    """
    1レース走らせて、(1) 外から見た映像 (2) 車が見ている画像と VAE の復元を並べた映像 を返す
    render_frames: list which is total_frames * numpy(height, width, 3)
    vision_frames: list which is total_frames * numpy(height, 2*width, 3)
    """
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    agent.reset()
    render_frames, vision_frames, total_reward, done = [], [], 0.0, False
    while not done:
        render_frames.append(env.render())                       # 外から見た画面 (400, 600, 3)
        x = preprocess(obs)                                       # 車が見ている 64x64 の画像
        mu, _ = vae.encoder(x)
        x_hat = vae.decoder(mu)                                   # VAE が z から復元した画像
        pair = torch.cat([x, x_hat], dim=3)[0]                    # shape: (img_channels, height, 2*width)
        vision_frames.append((pair.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
        total_reward += reward
        done = terminated or truncated
    env.close()
    return total_reward, render_frames, vision_frames

def main():
    with wandb.init(project="world-models-carracing", job_type="eval-controller", config=CONFIG) as run:
        cfg = run.config
        artifact = run.use_artifact(cfg["controller_artifact"])
        ckpt = load_checkpoint(os.path.join(artifact.download(), "controller.pt"))
        train_cfg = ckpt["config"]
        
        # Controller が学習したときと同じ VAE / MDN-RNN を使う（Artifact 経由なので依存関係も記録される）
        vae_path = os.path.join(run.use_artifact(train_cfg["vae_artifact"]).download(), "vae.pt")
        mdnrnn_path = os.path.join(run.use_artifact(train_cfg["mdnrnn_artifact"]).download(), "mdnrnn.pt")
        vae, mdnrnn = load_vae(vae_path), load_mdnrnn(mdnrnn_path)
        controller = Controller(latent_dim=vae.encoder.fc_mu.out_features, hidden_dim=mdnrnn.hidden_dim,
                                action_dim=mdnrnn.actions_dim, input_mode=train_cfg["input_mode"])
        controller.load_state_dict(ckpt["model_state_dict"])
        params = torch.nn.utils.parameters_to_vector(controller.parameters()).detach().numpy()
        run.config.update({"input_mode": train_cfg["input_mode"], "train_run": ckpt["wandb_run_path"]})

        # 1. n_episodes コースで平均スコア（並列）
        seeds = np.random.default_rng(cfg["seed"]).integers(0, 2**31 - 1, size=cfg["n_episodes"])
        ctx = mp.get_context("spawn")
        with ctx.Pool(cfg["num_workers"], initializer=init_worker,
                    initargs=(vae_path, mdnrnn_path, train_cfg["input_mode"], train_cfg["z_mode"])) as pool:
            rewards = pool.map(evaluate_params, [(params, [s]) for s in seeds])
        
        print(f"score over {cfg['n_episodes']} tracks: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
        run.summary["score_mean"] = float(np.mean(rewards))
        run.summary["score_std"] = float(np.std(rewards))
        run.log({"score_histogram": wandb.Histogram(rewards)})
        
        # 2. 映像（メインプロセスで、最初の n_videos コース）
        agent = WorldModelAgent(vae, mdnrnn, controller, train_cfg["z_mode"])
        for i, seed in enumerate(seeds[:cfg["n_videos"]]):
            total, render_frames, vision_frames = record_episode(agent, vae, int(seed))
            
            skip = cfg["video_frame_skip"]
            fps = 50 // skip   # CarRacing は 50 FPS
            run.log({
                f"video/drive_{i}": wandb.Video(to_video(render_frames[::skip], (300, 200)), fps=fps, format="mp4",
                                                caption=f"reward {total:.1f}"),
                f"video/vision_{i}": wandb.Video(to_video(vision_frames[::skip], (384, 192)), fps=fps, format="mp4",
                                                caption="left: what the car sees (64x64) / right: VAE reconstruction"),
            })
            print(f"video {i}: reward {total:.1f}, {len(render_frames)} frames")

if __name__ == "__main__":
    main()