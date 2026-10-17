import os
import pickle
import time
import multiprocessing as mp
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
import cma
import wandb
from agent import WorldModelAgent, load_vae, load_mdnrnn, rollout
from model.controller import Controller

CONFIG = {
    # 使うモデル
    "vae_artifact": "vae:v0",
    "mdnrnn_artifact": "mdnrnn:v1",
    "input_mode": "z_and_h",             # "z_and_h"（論文のフル）または "z_only"（比較用: V model only）
    "z_mode": "sample",              # "sample"（MDN-RNN の学習と同じ）または "mu"
    # CMA-ES
    "pop_size": 32,                  # 1世代の候補（個体）の数。論文は 64
    "trials_per_individual": 4,      # 1個体を何コース走らせて平均するか。論文は 16
    "sigma_init": 0.1,               # 最初の探索の広さ
    "num_generations": 200,
    # 評価
    "eval_every": 10,                # 何世代ごとに、分布の中心の重みを評価するか
    "eval_episodes": 32,             # 評価で走らせるコース数
    # 実行
    "num_workers": 10,               # 並列に走らせるプロセス数（CPU のコア数以下）
    "checkpoint_every": 5,           # 何世代ごとに CMA-ES の状態を保存するか
    "resume_from": None,             # 再開するときは "weights/controller_<run_id>_es.pkl"
    "seed": 0,
}

# ---- 並列実行用（各ワーカープロセスで1回だけモデルと環境を作る） ----
_worker = {}

def init_worker(vae_path, mdnrnn_path, input_mode, z_mode):
    torch.set_num_threads(1)   # 1プロセス1スレッド。プロセスどうしで CPU を取り合わないようにする
    vae = load_vae(vae_path)
    mdnrnn = load_mdnrnn(mdnrnn_path)
    controller = Controller(latent_dim=vae.encoder.fc_mu.out_features, hidden_dim=mdnrnn.hidden_dim,
                            action_dim=mdnrnn.actions_dim, input_mode=input_mode)
    _worker["controller"] = controller
    _worker["agent"] = WorldModelAgent(vae, mdnrnn, controller, z_mode)
    _worker["env"] = gym.make("CarRacing-v3")

def evaluate_params(args):
    # 1つの候補（重みのベクトル）を、seeds のコースで走らせて平均報酬を返す
    params, seeds = args
    torch.nn.utils.vector_to_parameters(torch.as_tensor(params, dtype=torch.float32),
                                        _worker["controller"].parameters())
    rewards = [rollout(_worker["agent"], _worker["env"], int(s)) for s in seeds]
    return float(np.mean(rewards))

# ---- ユーティリティ ----
def download(run, artifact_name, filename):
    return os.path.join(run.use_artifact(artifact_name).download(), filename)

def save_controller(path, controller, params, cfg, generation, metrics, run):
    torch.nn.utils.vector_to_parameters(torch.as_tensor(params, dtype=torch.float32), controller.parameters())
    torch.save({
        "model_state_dict": controller.state_dict(),
        "config": {k: v for k, v in dict(cfg).items() if not k.startswith("_")},
        "generation": generation,
        "metrics": metrics,
        "wandb_run_path": run.path,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }, path)

def main():
    with wandb.init(project="world-models-carracing", job_type="train-controller", config=CONFIG) as run:
        cfg = run.config
        vae_path = download(run, cfg["vae_artifact"], "vae.pt")
        mdnrnn_path = download(run, cfg["mdnrnn_artifact"], "mdnrnn.pt")
        
        # パラメータ数を数えるためだけに、メインプロセスでも1つ作る
        vae = load_vae(vae_path)
        mdnrnn = load_mdnrnn(mdnrnn_path)
        controller = Controller(latent_dim=vae.encoder.fc_mu.out_features, hidden_dim=mdnrnn.hidden_dim,
                                action_dim=mdnrnn.actions_dim, input_mode=cfg["input_mode"])
        n_params = sum(p.numel() for p in controller.parameters())
        run.config.update({"n_params": n_params, "num_cpus": os.cpu_count()})
        
        # CMA-ES の初期化、または途中から再開
        if cfg["resume_from"]:
            with open(cfg["resume_from"], "rb") as f:
                state = pickle.load(f)
            es, start_gen = state["es"], state["generation"] + 1
            best_eval, best_params, best_metrics = state["best_eval"], state["best_params"], state["best_metrics"]
        else:
            es = cma.CMAEvolutionStrategy(np.zeros(n_params), cfg["sigma_init"],
                {"popsize": cfg["pop_size"], "seed": cfg["seed"] + 1, "verbose": -9})
            start_gen, best_eval, best_params, best_metrics = 0, -float("inf"), None, None
        
        rng = np.random.default_rng(cfg["seed"] + start_gen)

        run.define_metric("generation")
        run.define_metric("fitness/*", step_metric="generation")
        run.define_metric("es/*", step_metric="generation")
        run.define_metric("time/*", step_metric="generation")
        run.define_metric("eval/mean", step_metric="generation", summary="max")
        run.define_metric("eval/std", step_metric="generation")

        os.makedirs("weights", exist_ok=True)
        best_path = f"weights/controller_{run.id}_best.pt"
        es_path = f"weights/controller_{run.id}_es.pkl"
        
        if best_params is not None:   # 再開したときは、前回までの最良の重みをこの run のファイル名で書き出しておく
            save_controller(best_path, controller, best_params, cfg, start_gen - 1, best_metrics, run)
        
        ctx = mp.get_context("spawn")
        
        with ctx.Pool(cfg["num_workers"], initializer=init_worker,
                    initargs=(vae_path, mdnrnn_path, cfg["input_mode"], cfg["z_mode"])) as pool:
            
            for gen in range(start_gen, cfg["num_generations"]):
                t0 = time.time()
                solutions = es.ask()   # pop_size 個の候補。それぞれ長さ n_params の numpy 配列
                # 同じ世代の候補は、すべて同じコースで走らせる（コースの当たり外れで差がつかないように）
                seeds = rng.integers(0, 2**31 - 1, size=cfg["trials_per_individual"])
                fitness = pool.map(evaluate_params, [(s, seeds) for s in solutions])
                es.tell(solutions, [-f for f in fitness])   # cma は最小化なので符号を反転
                
                log = {
                    "generation": gen,
                    "fitness/best": max(fitness),
                    "fitness/mean": float(np.mean(fitness)),
                    "fitness/worst": min(fitness),
                    "es/sigma": float(es.sigma),
                    "time/generation_sec": time.time() - t0,
                }

                if (gen + 1) % cfg["eval_every"] == 0 or gen == cfg["num_generations"] - 1:
                    center = es.result.xfavorite
                    eval_seeds = rng.integers(0, 2**31 - 1, size=cfg["eval_episodes"])
                    eval_rewards = pool.map(evaluate_params, [(center, [s]) for s in eval_seeds])
                    log["eval/mean"] = float(np.mean(eval_rewards))
                    log["eval/std"] = float(np.std(eval_rewards))
                    
                    if log["eval/mean"] > best_eval:
                        best_eval, best_params = log["eval/mean"], center.copy()
                        best_metrics = {"eval_mean": log["eval/mean"], "eval_std": log["eval/std"]}
                        save_controller(best_path, controller, best_params, cfg, gen, best_metrics, run)
                    
                run.log(log)
                print(f"gen {gen}: best {log['fitness/best']:.1f} mean {log['fitness/mean']:.1f} "
                    f"({log['time/generation_sec']:.0f}s)" + (f" eval {log['eval/mean']:.1f}" if "eval/mean" in log else ""))
                
                # 保存する
                if (gen + 1) % cfg["checkpoint_every"] == 0 or gen == cfg["num_generations"] - 1:
                    with open(es_path, "wb") as f:
                        pickle.dump({"es": es, "generation": gen, "best_eval": best_eval,
                                    "best_params": best_params, "best_metrics": best_metrics}, f)
                        
        artifact = wandb.Artifact("controller", type="model",metadata={**{k: v for k, v in dict(cfg).items() if not k.startswith("_")},
                                    "best_eval_mean": best_eval})
        artifact.add_file(best_path, name="controller.pt")
        run.log_artifact(artifact)

if __name__ == "__main__":
    main()