import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.normalize import Normalize, NormalizationMode
from lerobot.configs.types import PolicyFeature, FeatureType

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
        
        # Re-creating features dict with correct types
        self.features_map = {}
        for key, ft in self.dataset.features.items():
            dtype_str = ft["dtype"]
            if dtype_str in ["image", "video"]:
                f_type = FeatureType.VISUAL
            else:
                f_type = FeatureType.STATE
                
            self.features_map[key] = PolicyFeature(
                type=f_type,
                shape=ft["shape"]
            )

        # Norm Map
        norm_map = {
            "observation.image": NormalizationMode.MEAN_STD,
            "observation.state": NormalizationMode.MIN_MAX,
            "action": NormalizationMode.MIN_MAX
        }
        
        # Prepare Stats
        # Ensure image stats are present. If not, use ImageNet.
        if "observation.image" not in self.stats:
            # ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            self.stats["observation.image"] = {"mean": mean, "std": std}
            
        # Initialize Normalize
        self.normalize = Normalize(
            features=self.features_map,
            norm_map=norm_map,
            stats=self.stats
        )
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # Observation History
        start_obs_idx = idx - self.n_obs_steps + 1
        indices = list(range(start_obs_idx, idx + 1))
        current_episode = self.dataset[idx]["episode_index"]
        
        # Refine indices
        actual_indices = []
        first_valid_idx = -1
        
        for i in indices:
            if i >= 0 and self.dataset[i]["episode_index"] == current_episode:
                actual_indices.append(i)
                if first_valid_idx == -1:
                    first_valid_idx = i
            else:
                actual_indices.append(-1)
                
        final_indices = [i if i != -1 else first_valid_idx for i in actual_indices]
        
        # Fetch observations
        images = []
        agent_poses = []
        
        for i in final_indices:
            item = self.dataset[i]
            images.append(item["observation.image"].float() / 255.0) 
            agent_poses.append(item["observation.state"].float())
            
        # Stack -> (T, ...)
        image_stack = torch.stack(images) # (n_obs_steps, C, H, W)
        agent_pos_stack = torch.stack(agent_poses) # (n_obs_steps, D)
        
        # Action Chunking
        start_idx = idx
        end_idx = min(idx + self.chunk_len, len(self.dataset))
        
        raw_actions = self.dataset.hf_dataset[start_idx : end_idx]["action"]
        raw_actions = torch.stack([torch.as_tensor(a) for a in raw_actions])
        
        # Handle Episode Boundaries for Actions
        episode_indices = self.dataset.hf_dataset[start_idx : end_idx]["episode_index"]
        episode_indices = torch.tensor(episode_indices)
        
        mask = episode_indices != current_episode
        if mask.any():
            first_diff = mask.nonzero()
            if len(first_diff) > 0:
                idx_diff = first_diff[0].item()
                last_valid = raw_actions[idx_diff - 1] if idx_diff > 0 else torch.zeros(self.action_dim)
                raw_actions[idx_diff:] = last_valid
                
        if len(raw_actions) < self.chunk_len:
            pad_len = self.chunk_len - len(raw_actions)
            last_action = raw_actions[-1]
            padding = last_action.unsqueeze(0).repeat(pad_len, 1)
            raw_actions = torch.cat([raw_actions, padding], dim=0)
            
        # Normalize
        # Create a batch-like dict
        batch = {
            "observation.image": image_stack,
            "observation.state": agent_pos_stack,
            "action": raw_actions
        }
        
        batch = self.normalize(batch)
        
        return {
            "image": batch["observation.image"],
            "agent_pos": batch["observation.state"],
            "action": batch["action"].float()
        }
