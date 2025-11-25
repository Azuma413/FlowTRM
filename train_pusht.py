import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os

from models.recursive_reasoning.pusht_rf_trm import PushT_RF_TRM
from dataset.pusht_dataset import PushTDataset

def train():
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
        "forward_dtype": "float32"
    }
    
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
    train_dataset = PushTDataset("lerobot/pusht", split="train")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # Loop
    model.train()
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
                "epoch": epoch + 1
            })
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss}")
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})
        
    # Save
    os.makedirs("checkpoints/pusht", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pusht/model_final.pth")
    torch.save(model.state_dict(), "checkpoints/pusht/model_final.pth")
    print("Training finished.")
    wandb.finish()

if __name__ == "__main__":
    train()
