import torch
import gymnasium as gym
import gym_pusht
import numpy as np
import cv2
import os
from tqdm import tqdm

from models.recursive_reasoning.pusht_rf_trm import PushT_RF_TRM
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def get_stats(dataset_name="lerobot/pusht"):
    # Load dataset just to get stats
    dataset = LeRobotDataset(dataset_name)
    return dataset.meta.stats

def normalize_image(image):
    # image: (H, W, C) uint8 -> (1, C, H, W) float32 normalized
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    return image

def normalize_state(state, stats):
    # state: (D,) -> (1, D)
    # stats: dict
    state = torch.from_numpy(state).float().unsqueeze(0)
    
    if "min" in stats and "max" in stats:
        min_val = torch.tensor(stats["min"]).float()
        max_val = torch.tensor(stats["max"]).float()
        return 2 * (state - min_val) / (max_val - min_val) - 1
    return state

def unnormalize_action(action, stats):
    # action: (1, D) -> (D,)
    if "min" in stats and "max" in stats:
        min_val = torch.tensor(stats["min"]).float().to(action.device)
        max_val = torch.tensor(stats["max"]).float().to(action.device)
        action = (action + 1) / 2 * (max_val - min_val) + min_val
    return action.squeeze(0).cpu().numpy()

def evaluate_gym():
    # Config
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_path": "checkpoints/pusht/model_final.pth",
        "num_episodes": 1, # Run 1 episode for verification
        "max_steps": 300,
        "chunk_len": 16,
        "exec_steps": 8, # Receding Horizon Control
        
        # Model config (must match training)
        "batch_size": 1, # Required by config validation
        "action_dim": 2,
        "prop_dim": 2,
        "image_size": 96,
        "obs_dim": 512,
        "hidden_size": 512,
        "num_steps": 16,
        "num_heads": 8,
        "expansion": 4.0,
        "forward_dtype": "float32",
        
        # Legacy
        "seq_len": 0,
        "vocab_size": 0,
        "num_puzzle_identifiers": 0
    }
    
    device = config["device"]
    print(f"Using device: {device}")
    
    # Load Stats
    print("Loading stats...")
    stats = get_stats()["observation.state"]
    action_stats = get_stats()["action"]
    
    # Load Model
    model = PushT_RF_TRM(config)
    if os.path.exists(config["checkpoint_path"]):
        model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
        print(f"Loaded checkpoint from {config['checkpoint_path']}")
    else:
        print("Checkpoint not found! Using random weights (expect failure).")
        
    model.to(device)
    model.eval()
    
    # Env
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    
    # Loop
    for episode in range(config["num_episodes"]):
        obs, info = env.reset()
        done = False
        frames = []
        step = 0
        
        pbar = tqdm(total=config["max_steps"], desc=f"Episode {episode+1}")
        
        while not done and step < config["max_steps"]:
            # Prepare input
            # Obs: pixels (96, 96, 3), agent_pos (2,)
            image = obs["pixels"] # (96, 96, 3)
            agent_pos = obs["agent_pos"] # (2,)
            
            # Normalize
            img_tensor = normalize_image(image).to(device)
            state_tensor = normalize_state(agent_pos, stats).to(device)
            
            batch = {
                "image": img_tensor,
                "agent_pos": state_tensor
            }
            
            # Inference
            with torch.no_grad():
                carry = model.initial_carry(batch)
                carry, _, _, preds, _ = model(carry, batch)
                action_chunk = preds["action"] # (1, chunk_len, 2)
                
            # Execute chunk
            # Unnormalize actions
            # We need to unnormalize the whole chunk or just what we execute
            
            for k in range(config["exec_steps"]):
                if step >= config["max_steps"]: break
                
                action = action_chunk[:, k, :] # (1, 2)
                action_np = unnormalize_action(action, action_stats)
                
                # Step env
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                
                # Render
                frame = env.render()
                frames.append(frame)
                
                step += 1
                pbar.update(1)
                
                if done: break
                
        pbar.close()
        
        # Save video
        save_path = f"outputs/pusht_eval_ep{episode}.mp4"
        os.makedirs("outputs", exist_ok=True)
        
        # Use cv2 or imageio to save video
        # frames is list of (H, W, 3)
        if len(frames) > 0:
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(save_path, fourcc, 20, (width, height))
            
            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            cv2.destroyAllWindows()
            video.release()
            print(f"Saved video to {save_path}")
            
    env.close()

if __name__ == "__main__":
    evaluate_gym()
