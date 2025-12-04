import os
import warnings
import logging
# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision are deprecated.*")
# Suppress root logger warning about torchcodec
class TorchCodecFilter(logging.Filter):
    def filter(self, record):
        return "'torchcodec' is not available" not in record.getMessage()
logging.getLogger().addFilter(TorchCodecFilter())
import random
from dataclasses import dataclass, asdict
from typing import Optional, Any, Tuple
import argparse
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import gymnasium as gym
import numpy as np
from models.pusht_model import PushTModel
from dataset.pusht_dataset import PushTDataset
from eval_pusht_gym import run_eval_episode, get_dataset_info, setup_normalization

@dataclass
class TrainConfig:
    # Training
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    max_grad_norm: float = 1.0
    # Model
    action_dim: int = 2
    prop_dim: int = 2
    image_size: int = 96
    obs_dim: int = 512
    hidden_size: int = 512
    chunk_len: int = 16
    num_steps: int = 16
    num_heads: int = 8
    expansion: float = 4.0
    forward_dtype: str = "float32"
    n_obs_steps: int = 2
    # Evaluation & Logging
    eval_interval: int = 10
    eval_unit: str = "epochs"
    eval_max_steps: int = 300
    eval_exec_steps: int = 8
    save_interval: int = 50
    save_unit: str = "epochs"
    project_name: str = "flowtrm-pusht"
    checkpoint_dir: str = "checkpoints/pusht"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_training(config: TrainConfig) -> Tuple[PushTModel, torch.optim.Optimizer, DataLoader, gym.Env]:
    print(f"Using device: {config.device}")
    wandb.init(project=config.project_name, config=asdict(config))
    model = PushTModel(asdict(config))
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train_dataset = PushTDataset(
        "lerobot/pusht", 
        split="train",
        image_size=config.image_size,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        n_obs_steps=config.n_obs_steps
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    eval_env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    return model, optimizer, train_loader, eval_env

def train_epoch(
    model: PushTModel, 
    optimizer: torch.optim.Optimizer, 
    loader: DataLoader, 
    config: TrainConfig, 
    epoch: int, 
    global_step: int,
    eval_env: gym.Env,
    normalize: Any,
    unnormalize: Any
) -> int:
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    total_loss = 0.0
    for batch in pbar:
        batch = {k: v.to(config.device) for k, v in batch.items()}
        loss, metrics = model(batch)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        wandb.log({
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "epoch": epoch + 1,
            "global_step": global_step
        })
        if config.eval_unit == "steps" and global_step % config.eval_interval == 0 and global_step > 0:
            evaluate(model, eval_env, normalize, unnormalize, config, global_step)
            model.train()
        if config.save_unit == "steps" and global_step % config.save_interval == 0 and global_step > 0:
            save_checkpoint(model, config, global_step)
        global_step += 1
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss}")
    wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})
    return global_step

def evaluate(
    model: PushTModel, 
    env: gym.Env, 
    normalize: Any,
    unnormalize: Any,
    config: TrainConfig, 
    global_step: int
):
    model.eval()
    print(f"\nRunning evaluation at step {global_step}...")
    frames, timeout = run_eval_episode(
        model, env, normalize, unnormalize, config.device, 
        max_steps=config.eval_max_steps, 
        exec_steps=config.eval_exec_steps,
        n_obs_steps=config.n_obs_steps
    )
    if len(frames) > 0:
        frames_np = np.array(frames) # (T, H, W, C)
        frames_ch = frames_np.transpose(0, 3, 1, 2) # (T, C, H, W)
        wandb.log({
            "eval/video": wandb.Video(frames_ch, fps=20, format="mp4"),
            "global_step": global_step
        })

def save_checkpoint(model: PushTModel, config: TrainConfig, global_step: Optional[int] = None):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    if global_step is not None:
        save_path = os.path.join(config.checkpoint_dir, f"model_step_{global_step}.pth")
    else:
        save_path = os.path.join(config.checkpoint_dir, "model_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved checkpoint to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train PushT Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    base_config = OmegaConf.structured(TrainConfig)
    if os.path.exists(args.config):
        file_config = OmegaConf.load(args.config)
        config_node = OmegaConf.merge(base_config, file_config)
        config = OmegaConf.to_object(config_node)
        print(f"Loaded config from {args.config}")
    else:
        print(f"Config file {args.config} not found. Using defaults.")
        config = TrainConfig()
    set_seed(config.seed)
    model, optimizer, train_loader, eval_env = setup_training(config)
    stats, features = get_dataset_info()
    normalize, unnormalize = setup_normalization(stats, features)
    global_step = 0
    try:
        for epoch in range(config.epochs):
            global_step = train_epoch(
                model, optimizer, train_loader, config, epoch, global_step, eval_env, normalize, unnormalize
            )
            if config.eval_unit == "epochs" and (epoch + 1) % config.eval_interval == 0:
                evaluate(model, eval_env, normalize, unnormalize, config, global_step)
                model.train()
            if config.save_unit == "epochs" and (epoch + 1) % config.save_interval == 0:
                save_checkpoint(model, config, global_step)
        save_checkpoint(model, config)
        print("Training finished.")
    except KeyboardInterrupt:
        print("Training interrupted.")
        save_checkpoint(model, config, global_step)
    finally:
        wandb.finish()
        eval_env.close()

if __name__ == "__main__":
    main()
