
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.normalize import Normalize, Unnormalize, NormalizationMode
from lerobot.configs.types import PolicyFeature, FeatureType

def check_stats():
    print("Loading dataset...")
    dataset = LeRobotDataset("lerobot/pusht")
    stats = dataset.meta.stats
    features = dataset.features
    
    print("\n--- Stats Keys ---")
    print(stats.keys())
    
    if "observation.image" in stats:
        print("\n--- Observation Image Stats ---")
        print(stats["observation.image"])
    else:
        print("\nObservation Image Stats NOT found (will use ImageNet default in code)")

    if "action" in stats:
        print("\n--- Action Stats ---")
        print(stats["action"])
        
    # Simulate Setup Normalization
    features_map = {}
    for key, ft in features.items():
        dtype_str = ft["dtype"]
        if dtype_str in ["image", "video"]:
            f_type = FeatureType.VISUAL
        else:
            f_type = FeatureType.STATE
            
        features_map[key] = PolicyFeature(
            type=f_type,
            shape=ft["shape"]
        )

    norm_map = {
        "observation.image": NormalizationMode.MEAN_STD,
        "observation.state": NormalizationMode.MIN_MAX,
        "action": NormalizationMode.MIN_MAX
    }
    
    # Force ImageNet stats if requested (simulating the fix)
    # But let's see what happens if we don't force it first (current state)
    
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
    
    print("\n--- Normalization Check ---")
    # Create dummy action
    if "action" in stats:
        min_val = stats["action"]["min"]
        max_val = stats["action"]["max"]
        print(f"Action Min: {min_val}")
        print(f"Action Max: {max_val}")
        
        # Test normalization
        dummy_action = (min_val + max_val) / 2.0
        batch = {"action": dummy_action.unsqueeze(0)}
        norm_batch = normalize(batch)
        print(f"Normalized Action (Midpoint): {norm_batch['action']}")
        
        rec_batch = unnormalize(norm_batch)
        print(f"Reconstructed Action: {rec_batch['action']}")
        
        # Test bounds
        batch_min = {"action": min_val.unsqueeze(0)}
        norm_min = normalize(batch_min)
        print(f"Normalized Action (Min): {norm_min['action']}")
        
        batch_max = {"action": max_val.unsqueeze(0)}
        norm_max = normalize(batch_max)
        print(f"Normalized Action (Max): {norm_max['action']}")

if __name__ == "__main__":
    check_stats()
