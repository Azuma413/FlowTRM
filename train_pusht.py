import os
import random
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import gymnasium as gym
import gym_pusht  # Register PushT environments
import numpy as np

from models.recursive_reasoning.pusht_rf_trm import PushT_RF_TRM
from dataset.pusht_dataset import PushTDataset
from eval_pusht_gym import run_eval_episode, get_stats

@dataclass
class TrainConfig:
    # Training
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
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
    eval_freq_steps: int = 1000
    eval_max_steps: int = 300
    eval_exec_steps: int = 8
    save_freq_steps: int = 5000
    project_name: str = "flowtrm-pusht"
    checkpoint_dir: str = "checkpoints/pusht"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_training(config: TrainConfig) -> Tuple[PushT_RF_TRM, torch.optim.Optimizer, DataLoader, gym.Env]:
    # Device
    print(f"Using device: {config.device}")
    
    # WandB
    wandb.init(project=config.project_name, config=asdict(config))
    
    # Model
    model = PushT_RF_TRM(asdict(config))
    model.to(config.device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    # Data
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
    
    # Eval Env
    eval_env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    
    return model, optimizer, train_loader, eval_env

def train_epoch(
    model: PushT_RF_TRM, 
    optimizer: torch.optim.Optimizer, 
    loader: DataLoader, 
    config: TrainConfig, 
    epoch: int, 
    global_step: int,
    eval_env: gym.Env,
    stats: Dict[str, Any]
) -> int:
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    total_loss = 0.0
    
    for batch in pbar:
        # Move to device
        batch = {k: v.to(config.device) for k, v in batch.items()}
        
        # Forward
        loss, metrics = model(batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
        wandb.log({
            "train/loss": loss.item(),
            "epoch": epoch + 1,
            "global_step": global_step
        })
        
        # Evaluation
        if global_step % config.eval_freq_steps == 0 and global_step > 0:
            evaluate(model, eval_env, stats, config, global_step)
            model.train()
            
        # Checkpoint
        if global_step % config.save_freq_steps == 0 and global_step > 0:
            save_checkpoint(model, config, global_step)
        
        global_step += 1
        
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss}")
    wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})
    
    return global_step

def evaluate(
    model: PushT_RF_TRM, 
    env: gym.Env, 
    stats: Dict[str, Any], 
    config: TrainConfig, 
    global_step: int
):
    model.eval()
    print(f"\nRunning evaluation at step {global_step}...")
    
    frames, timeout = run_eval_episode(
        model, env, stats, config.device, 
        max_steps=config.eval_max_steps, 
        exec_steps=config.eval_exec_steps,
        n_obs_steps=config.n_obs_steps
    )
    
    # Log video
    if len(frames) > 0:
        frames_np = np.array(frames) # (T, H, W, C)
        frames_ch = frames_np.transpose(0, 3, 1, 2) # (T, C, H, W)
        wandb.log({
            "eval/video": wandb.Video(frames_ch, fps=20, format="mp4"),
            "global_step": global_step
        })

def save_checkpoint(model: PushT_RF_TRM, config: TrainConfig, global_step: Optional[int] = None):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    if global_step is not None:
        save_path = os.path.join(config.checkpoint_dir, f"model_step_{global_step}.pth")
    else:
        save_path = os.path.join(config.checkpoint_dir, "model_final.pth")
        
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved checkpoint to {save_path}")

def main():
    config = TrainConfig()
    set_seed(config.seed)
    
    model, optimizer, train_loader, eval_env = setup_training(config)
    stats = get_stats()
    
    global_step = 0
    
    try:
        for epoch in range(config.epochs):
            global_step = train_epoch(
                model, optimizer, train_loader, config, epoch, global_step, eval_env, stats
            )
            
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
