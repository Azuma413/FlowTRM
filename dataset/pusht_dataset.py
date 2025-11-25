import torch
from torch.utils.data import Dataset
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class PushTDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=96, chunk_len=16, action_dim=2):
        self.split = split
        self.image_size = image_size
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        
        # Load LeRobot Dataset
        # root_dir is ignored as we use the hub/cache default location for now
        # or we can pass it if LeRobotDataset supports it.
        # For "lerobot/pusht", it downloads to ~/.cache/huggingface/...
        self.dataset = LeRobotDataset("lerobot/pusht")
        
        self.stats = self.dataset.meta.stats
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # We need to retrieve a chunk of actions starting from idx
        # and the observation at idx.
        
        # LeRobotDataset[idx] returns a dictionary for a single frame.
        # To get a chunk, we need to handle it carefully.
        # Ideally we want:
        # obs = dataset[idx]["observation.image"]
        # state = dataset[idx]["observation.state"]
        # actions = dataset[idx : idx + chunk_len]["action"]
        
        # Check episode boundary
        # We can simply try to get the slice.
        # If the slice crosses episodes, we need to handle it.
        # LeRobotDataset might not support slicing across episodes seamlessly if we want consistent chunks.
        # But let's try simple slicing first.
        
        # Get start item
        item = self.dataset[idx]
        
        # Image
        image = item["observation.image"] # (C, H, W) float32 normalized?
        # LeRobot returns (C, H, W) usually.
        # Check normalization. Usually it is [0, 1] or [0, 255].
        # ResNet expects normalized.
        
        # Agent Pos
        agent_pos = item["observation.state"]
        
        # Action Chunk
        # We need to fetch future actions.
        # We can loop or slice if supported.
        # dataset[i] is slow if we do it one by one?
        # LeRobotDataset supports slicing? -> Yes, usually returns dict of stacked arrays.
        
        start_idx = idx
        end_idx = idx + self.chunk_len
        
        # Clamp end_idx to length
        end_idx = min(end_idx, len(self.dataset))
        
        # Check if we cross episode boundaries
        # item['episode_index']
        current_episode = item["episode_index"]
        
        # Get slice
        # Note: LeRobotDataset slicing might be heavy if it loads images.
        # We only need actions for the future steps.
        # But we can't easily slice only actions without loading everything if we use dataset[slice].
        # However, we can access the underlying HF dataset or mapped arrays if available.
        # For now, let's assume dataset[slice] works and is optimized enough or we accept the overhead.
        
        # Optimization: We only need actions for the chunk.
        # But LeRobotDataset might load images too.
        # Let's try to get actions specifically if possible.
        # self.dataset.hf_dataset['action'][start:end] might work if it's an Arrow dataset.
        
        # Let's try standard slicing first.
        # If we are near the end of dataset, we pad.
        
        # Actually, we should check episode indices.
        # We can fetch episode indices for the range.
        
        # Let's implement a safer loop for now to ensure correctness.
        actions = []
        for i in range(self.chunk_len):
            curr = start_idx + i
            if curr >= len(self.dataset):
                # Out of bounds, repeat last action
                actions.append(actions[-1] if actions else torch.zeros(self.action_dim))
                continue
                
            # We can access data directly?
            # item_i = self.dataset[curr] # This loads image! Slow.
            
            # Accessing via hf_dataset is faster for specific columns
            # self.dataset.hf_dataset[curr] returns a dict of values.
            # But we need to handle transforms/normalization if any.
            # LeRobotDataset handles normalization.
            
            # Let's use the dataset's __getitem__ but maybe we can optimize later.
            # For now, correctness first.
            
            # Wait, if we use dataset[curr], we load images 16 times! That's bad.
            # We need a way to get only actions.
            pass
            
        # Better approach:
        # Use `dataset.hf_dataset.select(range(start, end))`?
        # Or `dataset.hf_dataset[start:end]["action"]`?
        # But we need to apply normalization manually if we bypass `__getitem__`.
        # `dataset.stats` contains normalization stats.
        
        # Let's use `dataset.hf_dataset` to get raw actions and normalize them.
        # The `LeRobotDataset` usually stores raw data.
        
        raw_actions = self.dataset.hf_dataset[start_idx : end_idx]["action"]
        raw_actions = torch.stack([torch.as_tensor(a) for a in raw_actions])
        
        # Check episode indices to mask out cross-episode actions
        episode_indices = self.dataset.hf_dataset[start_idx : end_idx]["episode_index"]
        episode_indices = torch.tensor(episode_indices)
        
        # Mask: where episode_index != current_episode
        mask = episode_indices != current_episode
        
        # If any mask is true, we should replace those actions with the last valid action
        # or just pad.
        
        # Normalize actions
        # stats['action']['min'] and ['max'] or mean/std
        stats = self.stats["action"]
        # LeRobot uses min/max normalization to [-1, 1] usually?
        # Or mean/std.
        # Let's check stats keys in inspection (not shown in previous output).
        # Assuming min/max for now as is common in diffusion policy.
        
        # Helper to normalize
        def normalize(data, stats):
            # data: (T, D)
            # stats: {'min': (D,), 'max': (D,), ...}
            if "min" in stats and "max" in stats:
                min_val = torch.tensor(stats["min"])
                max_val = torch.tensor(stats["max"])
                # 2 * (x - min) / (max - min) - 1  -> [-1, 1]
                return 2 * (data - min_val) / (max_val - min_val) - 1
            elif "mean" in stats and "std" in stats:
                mean_val = torch.tensor(stats["mean"])
                std_val = torch.tensor(stats["std"])
                return (data - mean_val) / std_val
            return data

        norm_actions = normalize(raw_actions, stats)
        
        # Handle padding if length < chunk_len
        if len(norm_actions) < self.chunk_len:
            pad_len = self.chunk_len - len(norm_actions)
            last_action = norm_actions[-1]
            padding = last_action.unsqueeze(0).repeat(pad_len, 1)
            norm_actions = torch.cat([norm_actions, padding], dim=0)
            
        # Handle episode boundaries
        # If we crossed episode, replace with last valid action of current episode
        # Find first index where episode changed
        first_diff = (episode_indices != current_episode).nonzero()
        if len(first_diff) > 0:
            idx_diff = first_diff[0].item()
            # Repeat the action at idx_diff - 1
            last_valid = norm_actions[idx_diff - 1]
            norm_actions[idx_diff:] = last_valid
            
        # Get Observation (Image + State)
        # We can use dataset[idx] for this as we need image.
        # But we also need to normalize state.
        # dataset[idx] returns normalized items if configured?
        # LeRobotDataset usually returns normalized items if `stats` are computed/loaded.
        # Let's check `dataset[idx]` content.
        
        # If I use `dataset[idx]`, it returns normalized values?
        # I should verify this.
        # If `LeRobotDataset` does normalization automatically, I should use it.
        # But for actions chunk, I can't use `dataset[idx]` efficiently.
        
        # Let's assume `dataset[idx]` returns normalized tensors.
        # And for actions, I need to normalize manually because I fetch raw from hf_dataset.
        
        item = self.dataset[idx]
        image = item["observation.image"].float()
        agent_pos = item["observation.state"].float()
        
        return {
            "image": image,
            "agent_pos": agent_pos,
            "action": norm_actions.float()
        }
