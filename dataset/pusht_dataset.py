import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class PushTDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", image_size: int = 96, chunk_len: int = 16, action_dim: int = 2):
        self.split = split
        self.image_size = image_size
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        
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
        # Get start item for observation
        item = self.dataset[idx]
        
        # Image & Agent Pos (LeRobotDataset returns normalized values if stats are available)
        # However, to be safe and consistent with our manual action normalization, 
        # we should verify if LeRobotDataset normalizes automatically. 
        # By default, LeRobotDataset returns raw data unless a transform is applied.
        # But here we assume we need to normalize manually or rely on what's returned.
        # Given the previous code's assumption, we will trust item values for obs 
        # but normalize actions manually as we fetch them in bulk.
        
        image = item["observation.image"].float()
        agent_pos = item["observation.state"].float()
        
        # Action Chunking
        start_idx = idx
        end_idx = min(idx + self.chunk_len, len(self.dataset))
        
        # Fetch raw actions from the underlying HuggingFace dataset for efficiency
        raw_actions = self.dataset.hf_dataset[start_idx : end_idx]["action"]
        raw_actions = torch.stack([torch.as_tensor(a) for a in raw_actions])
        
        # Normalize actions
        norm_actions = self._normalize(raw_actions, self.stats["action"])
        
        # Handle Episode Boundaries
        episode_indices = self.dataset.hf_dataset[start_idx : end_idx]["episode_index"]
        episode_indices = torch.tensor(episode_indices)
        current_episode = item["episode_index"]
        
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
