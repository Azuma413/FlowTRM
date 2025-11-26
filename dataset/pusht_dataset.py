import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class PushTDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", image_size: int = 96, chunk_len: int = 16, action_dim: int = 2, n_obs_steps: int = 2):
        self.split = split
        self.image_size = image_size
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        
        # Load LeRobot Dataset
        self.dataset = LeRobotDataset("lerobot/pusht")
        self.stats = self.dataset.meta.stats
        
    def __len__(self):
        return len(self.dataset)
    
    def _normalize(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        if "min" in stats and "max" in stats:
            min_val = torch.tensor(stats["min"])
            max_val = torch.tensor(stats["max"])
            # Normalize to [-1, 1]
            return 2 * (data - min_val) / (max_val - min_val) - 1
        elif "mean" in stats and "std" in stats:
            mean_val = torch.tensor(stats["mean"])
            std_val = torch.tensor(stats["std"])
            return (data - mean_val) / std_val
        return data

    def __getitem__(self, idx: int):
        # Observation History
        # We need to fetch n_obs_steps observations ending at idx
        # i.e., [idx - n_obs_steps + 1, ..., idx]
        
        start_obs_idx = idx - self.n_obs_steps + 1
        
        # Handle indices < 0 or crossing episode boundaries
        # For simplicity, we'll just clamp to the start of the current episode
        # But LeRobotDataset is flat, so we need to be careful.
        # Actually, let's just fetch the indices and then fix them up.
        
        indices = list(range(start_obs_idx, idx + 1))
        
        # Get current episode index to check boundaries
        current_episode = self.dataset[idx]["episode_index"]
        
        valid_indices = []
        for i in indices:
            if i < 0:
                valid_indices.append(idx) # Placeholder, will be replaced
            elif self.dataset[i]["episode_index"] != current_episode:
                # If we crossed into previous episode, repeat the first valid frame of THIS episode
                # We can find the start of this episode, but for efficiency, 
                # let's just use the current index or the first valid index we found?
                # A better approach:
                # Find the first index in 'indices' that belongs to the current episode.
                # Then pad everything before it with that first index.
                valid_indices.append(i) # Temporarily add, we'll filter later
            else:
                valid_indices.append(i)
                
        # Refine indices
        # 1. Identify valid indices (same episode)
        actual_indices = []
        first_valid_idx = -1
        
        for i in indices:
            if i >= 0 and self.dataset[i]["episode_index"] == current_episode:
                actual_indices.append(i)
                if first_valid_idx == -1:
                    first_valid_idx = i
            else:
                actual_indices.append(-1) # Mark as invalid
                
        # 2. Replace invalid with first_valid_idx
        final_indices = [i if i != -1 else first_valid_idx for i in actual_indices]
        
        # Fetch observations
        images = []
        agent_poses = []
        
        for i in final_indices:
            item = self.dataset[i]
            images.append(item["observation.image"].float())
            agent_poses.append(item["observation.state"].float())
            
        # Stack -> (T, ...)
        image = torch.stack(images) # (n_obs_steps, C, H, W)
        agent_pos = torch.stack(agent_poses) # (n_obs_steps, D)
        
        # Action Chunking
        start_idx = idx
        end_idx = min(idx + self.chunk_len, len(self.dataset))
        
        # Fetch raw actions from the underlying HuggingFace dataset for efficiency
        raw_actions = self.dataset.hf_dataset[start_idx : end_idx]["action"]
        raw_actions = torch.stack([torch.as_tensor(a) for a in raw_actions])
        
        # Normalize actions
        norm_actions = self._normalize(raw_actions, self.stats["action"])
        
        # Handle Episode Boundaries for Actions
        episode_indices = self.dataset.hf_dataset[start_idx : end_idx]["episode_index"]
        episode_indices = torch.tensor(episode_indices)
        
        # Mask out actions from different episodes
        mask = episode_indices != current_episode
        if mask.any():
            # Find first index where episode changes
            first_diff = mask.nonzero()
            if len(first_diff) > 0:
                idx_diff = first_diff[0].item()
                # Repeat the last valid action
                last_valid = norm_actions[idx_diff - 1] if idx_diff > 0 else torch.zeros(self.action_dim)
                norm_actions[idx_diff:] = last_valid
                
        # Padding if chunk is shorter than required
        if len(norm_actions) < self.chunk_len:
            pad_len = self.chunk_len - len(norm_actions)
            last_action = norm_actions[-1]
            padding = last_action.unsqueeze(0).repeat(pad_len, 1)
            norm_actions = torch.cat([norm_actions, padding], dim=0)
            
        return {
            "image": image,
            "agent_pos": agent_pos,
            "action": norm_actions.float()
        }
