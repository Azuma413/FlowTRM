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
    Uses Prefix Conditioning: [x, t, y+z]
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
        
        # Projections
        self.action_proj = CastedLinear(config.action_dim, config.hidden_size, bias=True)
        self.obs_proj = CastedLinear(config.obs_dim, config.hidden_size, bias=True)
        
        # Dynamic z initialization from x
        self.x_to_z = nn.Sequential(
            CastedLinear(config.obs_dim, config.hidden_size, bias=True),
            nn.SiLU(),
            CastedLinear(config.hidden_size, config.chunk_len * config.hidden_size, bias=True)
        )
        
        # Transformer Block
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
        
        # Positional encoding for the sequence (y+z)
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
        
        # 1. Embeddings
        
        # Action embedding
        y_emb = self.action_proj(y_t) # (B, L, H)
        
        # Combine y and z (Information Integration)
        # "y + z" as the main sequence content
        content = y_emb + z_t + self.pos_emb # (B, L, H)
        
        # Time embedding
        if isinstance(t, float):
            t = torch.tensor(t, device=x.device, dtype=x.dtype).repeat(B)
        t_emb = self.time_emb(t.view(-1, 1).to(x.dtype)) # (B, H)
        t_token = t_emb.unsqueeze(1) # (B, 1, H)
        
        # Obs embedding
        x_emb = self.obs_proj(x).unsqueeze(1) # (B, 1, H)
        
        # 2. Prefix Conditioning
        # Sequence: [x_token, t_token, content_tokens]
        # Length: 1 + 1 + L
        seq = torch.cat([x_emb, t_token, content], dim=1) # (B, 2+L, H)
        
        # 3. Transformer Block
        h = seq
        
        h_norm = self.norm1(h)
        attn_out = self.attn(cos_sin=None, hidden_states=h_norm)
        h = h + attn_out
        
        h_norm = self.norm2(h)
        mlp_out = self.mlp(h_norm)
        h = h + mlp_out
        
        # 4. Extract Output
        # We only care about the outputs corresponding to the content tokens (last L tokens)
        out_content = h[:, 2:, :] # (B, L, H)
        
        # Outputs
        v_pred = self.to_v(out_content) # (B, L, action_dim)
        z_next = self.to_z(out_content) # (B, L, H)
        
        return v_pred, z_next

class RF_TRM_Inner(nn.Module):
    def __init__(self, config: RF_TRM_Config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        self.net = RF_TRM_Net(config)
        
        # Token embedding for discrete inputs (if used)
        self.token_emb = CastedEmbedding(config.vocab_size, config.obs_dim, init_std=0.02, cast_to=self.forward_dtype)
        
        # Adapter for discrete labels to action space (if used)
        self.emb_to_action = CastedLinear(config.obs_dim, config.action_dim, bias=True)
        
    def initialize_latent(self, obs_x):
        # Dynamic initialization from x
        # obs_x: (B, obs_dim)
        B = obs_x.shape[0]
        z_flat = self.net.x_to_z(obs_x) # (B, L*H)
        z = z_flat.view(B, self.config.chunk_len, self.config.hidden_size)
        return z

    def get_obs_embedding(self, inputs):
        # inputs: (B, seq_len) int32
        # Embed and pool to get (B, obs_dim)
        emb = self.token_emb(inputs.long()) # (B, S, D)
        # Mean pooling for simplicity to get global context x
        x = emb.mean(dim=1)
        return x

    def forward(self, carry: RF_TRM_InnerCarry, batch: Dict[str, torch.Tensor]):
        # This forward is used for INFERENCE/TRAINING via pretrain.py
        
        inputs = batch["inputs"]
        labels = batch["labels"]
        
        # Handle labels (discrete -> continuous)
        if labels.shape[1] > self.config.chunk_len:
            labels = labels[:, :self.config.chunk_len]
        elif labels.shape[1] < self.config.chunk_len:
            pad = self.config.chunk_len - labels.shape[1]
            labels = F.pad(labels, (0, pad), value=0)
            
        y_1_emb = self.token_emb(labels.long()) # (B, chunk_len, obs_dim)
        y_1_continuous = self.emb_to_action(y_1_emb)
        
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
        z = self.initialize_latent(obs_x) # Dynamic z
        
        total_loss = 0
        num_steps = self.config.num_steps
        
        for step in range(num_steps):
            t = step / num_steps
            
            # Target velocity (Rectified Flow: y_1 - y_0)
            target_v = y_1 - y_0
            
            # Input y_t (Conditional Flow Matching)
            t_val = t
            y_t_ideal = (1 - t_val) * y_0 + t_val * y_1
            
            # Predict
            v_pred, z_next = self.net(obs_x, y_t_ideal, z, t_val)
            
            # Loss
            loss_step = F.mse_loss(v_pred, target_v)
            total_loss += loss_step
            
            # Update z (Recurrent)
            z = z_next
            
        return total_loss / num_steps

    def inference(self, obs_x):
        B = obs_x.shape[0]
        device = obs_x.device
        
        y = torch.randn(B, self.config.chunk_len, self.config.action_dim, device=device, dtype=self.forward_dtype)
        z = self.initialize_latent(obs_x) # Dynamic z
        
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
