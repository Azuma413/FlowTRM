import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.normalize import Normalize, NormalizationMode
from lerobot.configs.types import PolicyFeature, FeatureType
from tqdm import tqdm

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

        # --- Cache Dataset in Memory ---
        print("Caching dataset in memory... (This may take a moment)")
        images = []
        agent_poses = []
        episode_indices = []
        
        for i in tqdm(range(len(self.dataset)), desc="Caching"):
            item = self.dataset[i]
            images.append(item["observation.image"].float() / 255.0)
            agent_poses.append(item["observation.state"].float())
            episode_indices.append(item["episode_index"])
            
        self.cached_images = torch.stack(images) # (N, C, H, W)
        self.cached_agent_poses = torch.stack(agent_poses) # (N, D)
        self.cached_episode_indices = torch.tensor(episode_indices) # (N,)
        
        # Cache Actions
        # Accessing via hf_dataset is faster for bulk access
        print("Caching actions...")
        self.cached_actions = torch.stack([torch.as_tensor(a) for a in self.dataset.hf_dataset["action"]]) # (N, action_dim)
        
        print("Dataset cached successfully.")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # Observation History
        start_obs_idx = idx - self.n_obs_steps + 1
        indices = torch.arange(start_obs_idx, idx + 1)
        
        # Handle out of bounds and episode boundaries efficiently
        # We can use the cached episode indices
        current_episode = self.cached_episode_indices[idx]
        
        # Clamp indices to be valid
        indices = torch.clamp(indices, min=0, max=len(self) - 1)
        
        # Check episode consistency
        episode_indices = self.cached_episode_indices[indices]
        mask = episode_indices == current_episode
        
        # If any index is from a different episode (or invalid before clamping), replace with the first valid index
        # But for observation history, we usually pad with the first frame of the episode.
        # Let's find the first valid index in the sequence that belongs to the current episode.
        # Actually, since we are looking back, the 'valid' frames are those that match current_episode.
        # If we cross boundary, we should repeat the first frame of the current episode.
        
        # Optimized logic:
        # 1. Get the valid indices
        valid_mask = episode_indices == current_episode
        if not valid_mask.all():
            # Find the first index that belongs to the current episode
            first_valid_pos = torch.where(valid_mask)[0][0] # First True position
            first_valid_idx = indices[first_valid_pos]
            # Replace invalid indices with the first valid one
            indices[~valid_mask] = first_valid_idx

        # Fetch observations from cache
        image_stack = self.cached_images[indices] # (n_obs_steps, C, H, W)
        agent_pos_stack = self.cached_agent_poses[indices] # (n_obs_steps, D)
        
        # Action Chunking
        start_idx = idx
        end_idx = min(idx + self.chunk_len, len(self))
        
        raw_actions = self.cached_actions[start_idx : end_idx]
        
        # Handle Episode Boundaries for Actions
        # If we cross into the next episode, we should pad with the last action of the current episode
        action_episode_indices = self.cached_episode_indices[start_idx : end_idx]
        mask = action_episode_indices != current_episode
        
        if mask.any():
            # Find where the next episode starts
            first_diff = torch.where(mask)[0]
            if len(first_diff) > 0:
                idx_diff = first_diff[0].item()
                # Repeat the last valid action
                last_valid_action = raw_actions[idx_diff - 1] if idx_diff > 0 else torch.zeros(self.action_dim)
                raw_actions = raw_actions.clone() # Avoid modifying cache
                raw_actions[idx_diff:] = last_valid_action
                
        # Pad if length is less than chunk_len (end of dataset)
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
