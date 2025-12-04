
import torch
import numpy as np
from dataset.pusht_dataset import PushTDataset
from eval_pusht_gym import get_dataset_info, setup_normalization

def check_changes():
    print("--- Checking PushTDataset ---")
    # Initialize dataset (this should trigger the stats overwrite)
    # We use a small subset or just init to check stats
    try:
        ds = PushTDataset(root_dir="lerobot/pusht", split="train")
        stats = ds.stats
        
        if "observation.image" in stats:
            mean = stats["observation.image"]["mean"]
            std = stats["observation.image"]["std"]
            print("PushTDataset observation.image stats:")
            print(f"Mean: {mean}")
            print(f"Std: {std}")
            
            expected_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            if torch.allclose(mean, expected_mean, atol=1e-4):
                print("SUCCESS: PushTDataset is using ImageNet mean.")
            else:
                print("FAILURE: PushTDataset is NOT using ImageNet mean.")
        else:
            print("FAILURE: observation.image not found in PushTDataset stats.")
            
    except Exception as e:
        print(f"Error checking PushTDataset: {e}")

    print("\n--- Checking eval_pusht_gym ---")
    try:
        stats, features = get_dataset_info()
        # setup_normalization modifies stats in place or uses them to create normalize
        # But wait, setup_normalization in my code takes stats and modifies a local copy or the passed dict?
        # Let's check the code. It modifies the passed 'stats' dict if I recall correctly, or creates a new one?
        # In eval_pusht_gym.py:
        # def setup_normalization(stats, features_dict):
        #    ...
        #    stats["observation.image"] = ...
        #    ...
        # So it modifies the passed stats object if it's a dict.
        
        normalize, unnormalize = setup_normalization(stats, features)
        
        # Check if stats were updated
        if "observation.image" in stats:
            mean = stats["observation.image"]["mean"]
            print("eval_pusht_gym stats after setup_normalization:")
            print(f"Mean: {mean}")
            
            expected_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            if torch.allclose(mean, expected_mean, atol=1e-4):
                print("SUCCESS: eval_pusht_gym is using ImageNet mean.")
            else:
                print("FAILURE: eval_pusht_gym is NOT using ImageNet mean.")
        else:
             print("FAILURE: observation.image not found in stats after setup_normalization.")

    except Exception as e:
        print(f"Error checking eval_pusht_gym: {e}")

if __name__ == "__main__":
    check_changes()
