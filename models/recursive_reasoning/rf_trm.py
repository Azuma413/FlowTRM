from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, CastedLinear, CastedEmbedding

@dataclass
class RF_TRM_InnerCarry:
    z: torch.Tensor # Latent thought

@dataclass
class RF_TRM_Carry:
    inner_carry: RF_TRM_InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor # Not really used for fixed steps, but kept for compatibility
    current_data: Dict[str, torch.Tensor]

class RF_TRM_Config(BaseModel):
    batch_size: int
    seq_len: int # Not strictly used if we have fixed chunk_len, but kept for compatibility
    vocab_size: int # For compatibility
    num_puzzle_identifiers: int # For compatibility
    
    # RF-TRM specific
    action_dim: int = 7
    chunk_len: int = 16
    obs_dim: int = 512
    hidden_size: int = 512
    num_steps: int = 16 # Number of flow steps
    
    forward_dtype: str = "bfloat16"
    
    # For compatibility with pretrain.py which might pass these
    puzzle_emb_ndim: int = 0
    expansion: float = 4.0
    num_heads: int = 8 # If using attention
    pos_encodings: str = "learned"
    rms_norm_eps: float = 1e-5
    
    # Loss config
    sigma_min: float = 1e-5 # For numerical stability

class RF_TRM_Net(nn.Module):
    """
    Single Tiny Network: (x, y_t, z_t, t) -> (v_pred, z_next)
    """
    def __init__(self, config: RF_TRM_Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Embeddings
        self.time_emb = nn.Sequential(
            CastedLinear(1, config.hidden_size, bias=True),
            nn.SiLU(),
            CastedLinear(config.hidden_size, config.hidden_size, bias=True)
        )
        
        # Input projection (concatenated x, y_t, z_t)
        # x: (B, obs_dim) -> (B, H)
        # y_t: (B, chunk_len, action_dim) -> flattened? or processed per token?
        # The user says "y_t: Noisy Action Chunk". 
        # Usually in Transformers, y_t is a sequence. 
        # But here the output is v_pred (same shape as y_t) and z_next.
        # Let's treat y_t as a sequence of tokens.
        
        # We need to process the sequence.
        # Let's use a simple Transformer Block or MLP Mixer.
        # User said "Single Tiny Network (2-layer)".
        
        # Let's assume we process everything as a sequence.
        # Sequence length = chunk_len.
        # x is global context. z is global context (or sequence?).
        # User says "z_t: Current thought state".
        
        # Let's define z as a vector (B, H) or sequence (B, L, H).
        # User: "z_t: Latent Reasoning... initial value is learnable parameter or embedding of x"
        # Let's make z a sequence of same length as y to allow detailed reasoning, or just a vector.
        # Given "Action Chunk", maybe z is also a chunk.
        
        # Let's assume z is (B, chunk_len, H).
        
        self.action_proj = CastedLinear(config.action_dim, config.hidden_size, bias=True)
        self.obs_proj = CastedLinear(config.obs_dim, config.hidden_size, bias=True)
        
        # We will concatenate inputs or add them?
        # Flow Matching usually adds time embedding.
        
        # Processing block
        # Simple 2-layer MLP per token + mixing? Or Transformer?
        # User referenced "Single Network, 2-layer".
        # Let's use a Transformer Block for mixing information across time (chunk).
        
        from models.layers import Attention
        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(config.hidden_size, expansion=config.expansion)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, config.chunk_len, config.hidden_size) * 0.02)
        
        # Output heads
        self.to_v = CastedLinear(config.hidden_size, config.action_dim, bias=True)
        self.to_z = CastedLinear(config.hidden_size, config.hidden_size, bias=True) # Update z
        
    def forward(self, x, y_t, z_t, t):
        # x: (B, obs_dim)
        # y_t: (B, L, action_dim)
        # z_t: (B, L, H)
        # t: (B,) or scalar
        
        B, L, _ = y_t.shape
        
        # Embeddings
        y_emb = self.action_proj(y_t) # (B, L, H)
        y_emb = y_emb + self.pos_emb
        
        # Time embedding
        if isinstance(t, float):
            t = torch.tensor(t, device=x.device, dtype=x.dtype).repeat(B)
        t_emb = self.time_emb(t.view(-1, 1).to(x.dtype)) # (B, H)
        t_emb = t_emb.unsqueeze(1) # (B, 1, H)
        
        # Obs embedding
        x_emb = self.obs_proj(x).unsqueeze(1) # (B, 1, H)
        
        # Combine inputs
        # h = y_emb + z_t + t_emb + x_emb
        h = y_emb + z_t + t_emb + x_emb
        
        # Transformer Block
        # Self-attention
        # We need cos_sin for RoPE if used, but let's stick to simple learned pos enc or no pos enc for now if not passed.
        # The Attention layer expects cos_sin.
        # Let's pass None for cos_sin for now (no RoPE) or implement it.
        # User said "Positional Encoding... add to y".
        
        h_norm = self.norm1(h)
        attn_out = self.attn(cos_sin=None, hidden_states=h_norm)
        h = h + attn_out
        
        h_norm = self.norm2(h)
        mlp_out = self.mlp(h_norm)
        h = h + mlp_out
        
        # Outputs
        v_pred = self.to_v(h) # (B, L, action_dim)
        z_next = self.to_z(h) # (B, L, H)
        
        return v_pred, z_next

class RF_TRM_Inner(nn.Module):
    def __init__(self, config: RF_TRM_Config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        self.net = RF_TRM_Net(config)
        
        # Learnable initial z
        self.z_init = nn.Parameter(torch.randn(1, config.chunk_len, config.hidden_size) * 0.02)
        
        # Placeholder for obs encoder if input is raw features
        # Assuming input is already features for now, or we project it.
        # If input is token IDs (from PuzzleDataset), we need an embedding.
        # But user said "Observation (x): Image features...".
        # For compatibility with PuzzleDataset (tokens), let's add an embedding layer
        # and pool it to get obs_dim? Or just use the embedding as x?
        
        self.token_emb = CastedEmbedding(config.vocab_size, config.obs_dim, init_std=0.02, cast_to=self.forward_dtype)
        
    def initialize_latent(self, batch_size, device):
        return self.z_init.expand(batch_size, -1, -1).to(device).to(self.forward_dtype)

    def get_obs_embedding(self, inputs):
        # inputs: (B, seq_len) int32
        # Embed and pool to get (B, obs_dim)
        emb = self.token_emb(inputs.long()) # (B, S, D)
        # Mean pooling for simplicity to get global context x
        x = emb.mean(dim=1)
        return x

    def forward(self, carry: RF_TRM_InnerCarry, batch: Dict[str, torch.Tensor]):
        # This forward is used for INFERENCE (one step of the outer loop?)
        # Or is it the whole process?
        # In TRM, forward does the whole recursion.
        
        # But here we have "Training via Unrolled Flow Matching".
        # So we should do the whole loop in forward.
        
        # However, pretrain.py expects:
        # carry, loss, metrics, preds, all_finish = model(carry, batch)
        
        # If we are training, we compute loss.
        # If we are evaluating, we might want to return predictions.
        
        inputs = batch["inputs"]
        # Assuming labels are the target actions (y_1)
        # We need to embed labels to continuous space if they are tokens.
        # Let's use the same token_emb for labels? Or a separate one?
        # If labels are tokens, we can't do flow matching directly unless we embed them.
        # Let's assume we embed them.
        
        labels = batch["labels"]
        # labels: (B, L)
        # We need to resize labels to chunk_len if different.
        # Let's assume they match or we crop/pad.
        
        # For the sake of this implementation, let's assume we use the first chunk_len tokens.
        if labels.shape[1] > self.config.chunk_len:
            labels = labels[:, :self.config.chunk_len]
        elif labels.shape[1] < self.config.chunk_len:
            # Pad
            pad = self.config.chunk_len - labels.shape[1]
            labels = F.pad(labels, (0, pad), value=0)
            
        # Target y_1
        y_1 = self.token_emb(labels.long()) # (B, chunk_len, obs_dim)
        # We need to project to action_dim? Or set action_dim = obs_dim?
        # Let's project.
        # Actually, if we want to reconstruct tokens, we might need a head.
        # But user asked for Flow Matching on "Action Chunk".
        # Let's project y_1 to action_dim.
        # We need a way to map back to tokens for evaluation if we want to use existing eval.
        # But let's stick to the user's "Action Chunk" definition.
        
        # Project embedded labels to action space
        # We need an adapter to map embedding to action_dim
        # Let's add it to __init__
        if not hasattr(self, "emb_to_action"):
             self.emb_to_action = CastedLinear(self.config.obs_dim, self.config.action_dim, bias=True)
        
        y_1_continuous = self.emb_to_action(y_1)
        
        obs_x = self.get_obs_embedding(inputs)
        
        if self.training:
            return self.training_step(obs_x, y_1_continuous)
        else:
            return self.inference(obs_x)

    def training_step(self, obs_x, y_1):
        # Unrolled Flow Matching
        B = obs_x.shape[0]
        device = obs_x.device
        
        y_0 = torch.randn_like(y_1)
        z = self.initialize_latent(B, device)
        
        total_loss = 0
        
        # Add positional encoding to y_0 and y_1?
        # User: "y (軌道) をTransformerに入力する際、Chunk内の時間順序を示す1D Positional Encodingを加算"
        # We should add it inside the loop or before.
        # Since y changes, maybe add it every time we input y.
        
        # Loop
        num_steps = self.config.num_steps
        
        for step in range(num_steps):
            t = step / num_steps
            
            # Target velocity (Rectified Flow: y_1 - y_0)
            target_v = y_1 - y_0
            
            # Input y_t for stability: interpolation + noise?
            # User recommended: "Input y_t is ideal interpolation + small noise"
            # y_t_input = (1 - t) * y_0 + t * y_1
            # But we also want to simulate the recursive update?
            # User: "zのみ再帰的に伝播させるハイブリッド方式が良い"
            
            t_val = t
            y_t_ideal = (1 - t_val) * y_0 + t_val * y_1
            # Add small noise?
            # sigma = 0.1?
            # Let's just use y_t_ideal for now as per "Conditional Flow Matching"
            
            v_pred, z_next = self.net(obs_x, y_t_ideal, z, t_val)
            
            # Loss
            loss_step = F.mse_loss(v_pred, target_v)
            total_loss += loss_step
            
            # Update z
            z = z_next
            
        return total_loss / num_steps

    def inference(self, obs_x):
        B = obs_x.shape[0]
        device = obs_x.device
        
        y = torch.randn(B, self.config.chunk_len, self.config.action_dim, device=device, dtype=self.forward_dtype)
        z = self.initialize_latent(B, device)
        
        dt = 1.0 / self.config.num_steps
        
        for step in range(self.config.num_steps):
            t = step / self.config.num_steps
            
            v_pred, z_next = self.net(obs_x, y, z, t)
            
            y = y + v_pred * dt
            z = z_next
            
        return y

class RF_TRM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = RF_TRM_Config(**config_dict)
        self.inner = RF_TRM_Inner(self.config)
        
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return RF_TRM_Carry(
            inner_carry=RF_TRM_InnerCarry(z=torch.empty(0)), # Placeholder
            steps=torch.zeros(batch_size, dtype=torch.int32),
            halted=torch.zeros(batch_size, dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: RF_TRM_Carry, batch: Dict[str, torch.Tensor], return_keys=None):
        # This is called by pretrain.py
        # return: carry, loss, metrics, preds, all_finish
        
        if self.training:
            loss = self.inner(carry.inner_carry, batch)
            metrics = {"loss": loss.detach()}
            return carry, loss, metrics, {}, True
        else:
            # Inference
            inputs = batch["inputs"]
            obs_x = self.inner.get_obs_embedding(inputs)
            y_pred = self.inner.inference(obs_x)
            
            # We need to return something that looks like logits or predictions
            # If the evaluator expects tokens, we might need to map back.
            # But for now, let's return the continuous output.
            preds = {"actions": y_pred}
            
            return carry, torch.tensor(0.0), {}, preds, True
