from datetime import datetime
import torch


def save_checkpoint(path, model, optimizer, config, epoch, metrics, run=None):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(config),
        "epoch": epoch,
        "metrics": metrics,                         # 例: {"train_loss": 1.23, "val_loss": 1.31}
        "wandb_run_path": run.path if run else None,  # "entity/project/run_id"。コミットや差分はこの run に記録されている
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }, path)


def load_checkpoint(path, map_location="cpu"):
    # weights_only=True: dict / tensor / 数値 / 文字列だけを読み込む（任意コードを実行しない安全な読み込み）
    return torch.load(path, map_location=map_location, weights_only=True)
