import torch
from models.recursive_reasoning.rf_trm import RF_TRM
from models.losses import FlowMatchingLossHead

def test_rf_trm():
    config = {
        "batch_size": 2,
        "seq_len": 32,
        "vocab_size": 100,
        "num_puzzle_identifiers": 1,
        "action_dim": 7,
        "chunk_len": 16,
        "obs_dim": 64,
        "hidden_size": 64,
        "num_steps": 4,
        "expansion": 4,
        "num_heads": 4,
        "puzzle_emb_ndim": 0,
        "pos_encodings": "learned",
        "forward_dtype": "float32" # Use float32 for test
    }
    
    model = RF_TRM(config)
    print("Model instantiated")
    
    # Test initial carry
    batch = {
        "inputs": torch.randint(0, 100, (2, 32)),
        "labels": torch.randint(0, 100, (2, 32)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long)
    }
    carry = model.initial_carry(batch)
    print("Initial carry created")
    
    # Test training forward
    model.train()
    carry, loss, metrics, preds, all_finish = model(carry, batch)
    print(f"Training loss: {loss.item()}")
    assert loss.item() >= 0
    
    # Test inference forward
    model.eval()
    carry = model.initial_carry(batch)
    carry, loss, metrics, preds, all_finish = model(carry, batch)
    print("Inference completed")
    assert "actions" in preds
    print(f"Prediction shape: {preds['actions'].shape}")
    assert preds['actions'].shape == (2, 16, 7)

    # Test Loss Head
    loss_head = FlowMatchingLossHead(model)
    carry, loss, metrics, preds, all_finish = loss_head(return_keys=[], carry=carry, batch=batch)
    print("Loss head forward completed")

if __name__ == "__main__":
    test_rf_trm()
