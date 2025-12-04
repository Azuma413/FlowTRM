import torch
import gymnasium as gym
import gym_pusht
import numpy as np
import cv2
import os
from collections import deque
from tqdm import tqdm

from models.pusht_model import PushTModel
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.normalize import Normalize, Unnormalize, NormalizationMode
from lerobot.configs.types import PolicyFeature, FeatureType

def get_dataset_info(dataset_name="lerobot/pusht"):
    # Load dataset to get stats and features
    dataset = LeRobotDataset(dataset_name)
    stats = dataset.meta.stats
    features = dataset.features
    return stats, features

def setup_normalization(stats, features_dict):
    # Re-creating features dict with correct types
    features_map = {}
    for key, ft in features_dict.items():
        dtype_str = ft["dtype"]
        if dtype_str in ["image", "video"]:
            f_type = FeatureType.VISUAL
        else:
            f_type = FeatureType.STATE
            
        shape = ft["shape"]
        if f_type == FeatureType.VISUAL and len(shape) == 3 and shape[-1] == 3:
            shape = (shape[2], shape[0], shape[1])
            
        features_map[key] = PolicyFeature(
            type=f_type,
            shape=shape
        )

    # Norm Map
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.MIN_MAX,
        FeatureType.ACTION: NormalizationMode.MIN_MAX
    }
    
    # Prepare Stats
    # Prepare Stats
    # Force ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    stats["observation.image"] = {"mean": mean, "std": std}
    
    # Convert all stats to Tensors
    for k, v in stats.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, np.ndarray):
                    v[sk] = torch.from_numpy(sv)
        
    normalize = Normalize(
        features=features_map,
        norm_map=norm_map,
        stats=stats
    )
    
    unnormalize = Unnormalize(
        features=features_map,
        norm_map=norm_map,
        stats=stats
    )
    
    return normalize, unnormalize

def run_eval_episode(model, env, normalize, unnormalize, device, max_steps=300, exec_steps=8, n_obs_steps=2):
    obs, info = env.reset()
    done = False
    frames = []
    step = 0
    # History buffers
    obs_history = {
        "image": deque(maxlen=n_obs_steps),
        "agent_pos": deque(maxlen=n_obs_steps)
    }
    
    while not done and step < max_steps:
        # Prepare input
        image = obs["pixels"] # (96, 96, 3)
        agent_pos = obs["agent_pos"] # (2,)
        # To Tensor
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0 # (3, 96, 96) [0,1]
        state_tensor = torch.from_numpy(agent_pos).float() # (2,)
        # Update history
        if len(obs_history["image"]) == 0:
            for _ in range(n_obs_steps):
                obs_history["image"].append(img_tensor)
                obs_history["agent_pos"].append(state_tensor)
        else:
            obs_history["image"].append(img_tensor)
            obs_history["agent_pos"].append(state_tensor)
        # Stack history -> (T, ...)
        img_hist = torch.stack(list(obs_history["image"])) # (T, C, H, W)
        pos_hist = torch.stack(list(obs_history["agent_pos"])) # (T, D)
        batch = {
            "observation.image": img_hist,
            "observation.state": pos_hist
        }
        batch = {k: v.to(device) for k, v in batch.items()}
        normalize.to(device)
        batch = normalize(batch)
        model_input = {
            "image": batch["observation.image"].unsqueeze(0),
            "agent_pos": batch["observation.state"].unsqueeze(0)
        }
        # Warm Start Logic
        initial_guess = None
        if "prev_action_chunk" in locals() and prev_action_chunk is not None:
            # Shift previous action chunk by exec_steps
            # prev_action_chunk: (1, chunk_len, action_dim)
            # We need to shift left by exec_steps and pad
            
            # Note: action_chunk is normalized.
            # We assume the model expects normalized initial_guess.
            
            prev_chunk = prev_action_chunk.squeeze(0) # (chunk_len, action_dim)
            chunk_len = prev_chunk.shape[0]
            
            if exec_steps < chunk_len:
                new_guess = torch.zeros_like(prev_chunk)
                new_guess[:-exec_steps] = prev_chunk[exec_steps:]
                new_guess[-exec_steps:] = prev_chunk[-1] # Pad with last action
                initial_guess = new_guess.unsqueeze(0) # (1, chunk_len, action_dim)
            else:
                # If exec_steps >= chunk_len, we can't really warm start meaningfully from this chunk alone
                # But let's just replicate the last action
                initial_guess = prev_chunk[-1].unsqueeze(0).repeat(chunk_len, 1).unsqueeze(0)
        else:
            # First step or no history: Use current agent_pos as initial guess
            # agent_pos is (2,) numpy array
            # We need to normalize it to use as initial guess for action.
            # We treat it as an action for normalization purposes.
            
            pos_tensor = torch.from_numpy(agent_pos).float().to(device).unsqueeze(0) # (1, 2)
            dummy_batch = {"action": pos_tensor}
            normalize.to(device)
            norm_batch = normalize(dummy_batch)
            norm_pos = norm_batch["action"] # (1, 2)
            
            chunk_len = model.irm.chunk_size
            initial_guess = norm_pos.unsqueeze(1).repeat(1, chunk_len, 1) # (1, chunk_len, 2)

        # Inference
        with torch.no_grad():
            preds = model.predict(model_input, initial_guess=initial_guess)
            action_chunk = preds["action"] # (1, chunk_len, 2)
            prev_action_chunk = action_chunk.clone()
            
        # Execute chunk
        # Action chunk is normalized. We need to unnormalize.
        # Unnormalize expects batch.
        
        # We can unnormalize the whole chunk.
        action_batch = {
            "action": action_chunk.squeeze(0) # (chunk_len, 2)
        }
        unnormalize.to(device)
        action_batch = unnormalize(action_batch)
        action_chunk_np = action_batch["action"].cpu().numpy()
        
        for k in range(exec_steps):
            if step >= max_steps: break
            action = action_chunk_np[k] # (2,)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frame = env.render()
            frames.append(frame)
            step += 1
            if done: break
    return frames, step >= max_steps

def evaluate_gym():
    # Config
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_path": "checkpoints/pusht/model_step_5050.pth", # Default to final
        "num_episodes": 1,
        "max_steps": 300,
        "chunk_len": 16,
        "exec_steps": 8,
        
        # Model config
        "batch_size": 1,
        "action_dim": 2,
        "prop_dim": 2,
        "image_size": 96,
        "obs_dim": 512,
        "hidden_size": 512,
        "num_steps": 16,
        "num_heads": 8,
        "expansion": 4.0,
        "forward_dtype": "float32",
        "n_obs_steps": 2,
    }
    
    device = config["device"]
    print(f"Using device: {device}")
    
    # Load Stats & Features
    print("Loading stats and features...")
    stats, features = get_dataset_info()
    normalize, unnormalize = setup_normalization(stats, features)
    
    # Load Model
    model = PushTModel(config)
    
    ckpt_path = config["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        # Try to find latest checkpoint
        ckpt_dir = os.path.dirname(ckpt_path)
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth") and "model_step_" in f]
            if len(ckpts) > 0:
                ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
                print(f"model_final.pth not found, using latest: {ckpt_path}")
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("Checkpoint not found! Using random weights.")
        
    model.to(device)
    model.eval()
    
    # Env
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    
    # Loop
    for episode in range(config["num_episodes"]):
        print(f"Episode {episode+1}")
        frames, timeout = run_eval_episode(
            model, env, normalize, unnormalize, device, 
            config["max_steps"], config["exec_steps"], config["n_obs_steps"]
        )
        
        # Save video
        save_path = f"outputs/pusht_eval_ep{episode}.mp4"
        os.makedirs("outputs", exist_ok=True)
        
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
