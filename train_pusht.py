import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import gymnasium as gym
import gym_pusht
import numpy as np

from models.recursive_reasoning.pusht_rf_trm import PushT_RF_TRM
from dataset.pusht_dataset import PushTDataset
from eval_pusht_gym import run_eval_episode, get_stats

def train(config_updates=None):
    # Config
    config = {
        "batch_size": 16,
        "lr": 1e-4,
        "epochs": 500,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        # Model config
        "action_dim": 2,
        "prop_dim": 2,
        "image_size": 96,
        "obs_dim": 512,
        "hidden_size": 512,
        "chunk_len": 16,
        "num_steps": 16,
        "num_heads": 8,
        "expansion": 4.0,
        "forward_dtype": "float32",
        
        # Eval config
        "eval_freq_steps": 1000, # Run eval every N steps
        "eval_max_steps": 300,
        "eval_exec_steps": 8,
        "save_freq_steps": 5000, # Save checkpoint every N steps
    }
    
    if config_updates:
        config.update(config_updates)
    
    
    # Setup
    device = config["device"]
    print(f"Using device: {device}")

    # WandB
    wandb.init(project="flowtrm-pusht", config=config)
    
    # Model
    model = PushT_RF_TRM(config)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    # Data
    # root_dir is ignored by LeRobotDataset but we pass it for compatibility
    train_dataset = PushTDataset(
        "lerobot/pusht", 
        split="train",
        image_size=config["image_size"],
        chunk_len=config["chunk_len"],
        action_dim=config["action_dim"]
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # Eval Setup
    eval_env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    stats = get_stats()
    
    # Loop
    model.train()
    global_step = 0
    
    os.makedirs("checkpoints/pusht", exist_ok=True)
    
    for epoch in range(config["epochs"]):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        total_loss = 0
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward
            # We need a dummy carry for compatibility or just pass None if we modify forward
            # The current forward expects carry. Let's create one.
            carry = model.initial_carry(batch)
            
            carry, loss, metrics, preds, all_finish = model(carry, batch)
            
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
            if global_step % config["eval_freq_steps"] == 0 and global_step > 0:
                model.eval()
                print(f"\nRunning evaluation at step {global_step}...")
                frames, timeout = run_eval_episode(
                    model, eval_env, stats, device, 
                    max_steps=config["eval_max_steps"], 
                    exec_steps=config["eval_exec_steps"]
                )
                
                # Log video
                if len(frames) > 0:
                    frames_np = np.array(frames) # (T, H, W, C)
                    frames_ch = frames_np.transpose(0, 3, 1, 2) # (T, C, H, W)
                    wandb.log({
                        "eval/video": wandb.Video(frames_ch, fps=20, format="mp4"),
                        "global_step": global_step
                    })
                
                model.train()
                
            # Checkpoint
            if global_step % config["save_freq_steps"] == 0 and global_step > 0:
                save_path = f"checkpoints/pusht/model_step_{global_step}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"\nSaved checkpoint to {save_path}")
            
            global_step += 1
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss}")
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})
        
    # Save Final
    torch.save(model.state_dict(), "checkpoints/pusht/model_final.pth")
    print("Training finished.")
    wandb.finish()
    eval_env.close()

if __name__ == "__main__":
    train()
