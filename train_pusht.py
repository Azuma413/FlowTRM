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
        "epochs": 10,
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
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")
        
    # Save
    os.makedirs("checkpoints/pusht", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pusht/model_final.pth")
    print("Training finished.")

if __name__ == "__main__":
    train()
