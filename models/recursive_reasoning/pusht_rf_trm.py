from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from models.layers import CastedLinear
from models.recursive_reasoning.rf_trm import RF_TRM_Net, RF_TRM_Config, RF_TRM_Carry, RF_TRM_InnerCarry
from models.vision import ResNet18Encoder

class PushT_RF_TRM_Config(RF_TRM_Config):
    # PushT specific defaults
    action_dim: int = 2 # PushT has 2D action (x, y)
    prop_dim: int = 2 # Agent pos (x, y)
    image_size: int = 96
    
    # Legacy fields (not used for PushT but required by base class)
    seq_len: int = 0
    vocab_size: int = 0
    num_puzzle_identifiers: int = 0
    
    # Obs encoder
    vision_backbone: str = "resnet18"
    pretrained_vision: bool = True
    freeze_vision: bool = False
    
    # Overwrite defaults
    obs_dim: int = 512 # Combined embedding size
    chunk_len: int = 16
    num_steps: int = 16

class PushT_RF_TRM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = PushT_RF_TRM_Config(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        # Components
        self.vision_encoder = ResNet18Encoder(
            pretrained=self.config.pretrained_vision,
            freeze=self.config.freeze_vision,
            output_dim=self.config.obs_dim // 2 # Allocate half for vision
        )
        
        self.prop_encoder = nn.Sequential(
            CastedLinear(self.config.prop_dim, self.config.obs_dim // 2, bias=True),
            nn.SiLU(),
            CastedLinear(self.config.obs_dim // 2, self.config.obs_dim // 2, bias=True)
        )
        
        # Fusion: We concatenate vision and prop embeddings -> obs_dim
        # obs_dim must match config.obs_dim
        
        # Core Network
        self.net = RF_TRM_Net(self.config)
        
        # Learnable initial z
        self.z_init = nn.Parameter(torch.randn(1, self.config.chunk_len, self.config.hidden_size) * 0.02)
        
    def initialize_latent(self, batch_size, device):
        return self.z_init.expand(batch_size, -1, -1).to(device).to(self.forward_dtype)

    def get_obs_embedding(self, images, agent_pos):
        # images: (B, C, H, W)
        # agent_pos: (B, prop_dim)
        
        img_emb = self.vision_encoder(images) # (B, obs_dim//2)
        prop_emb = self.prop_encoder(agent_pos) # (B, obs_dim//2)
        
        # Concatenate
        obs_x = torch.cat([img_emb, prop_emb], dim=-1) # (B, obs_dim)
        return obs_x

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        # Placeholder for compatibility with pretrain.py loop if needed
        # But for PushT we might write a custom loop.
        # If we use pretrain.py, we need this.
        batch_size = batch["image"].shape[0]
        return RF_TRM_Carry(
            inner_carry=RF_TRM_InnerCarry(z=torch.empty(0)),
            steps=torch.zeros(batch_size, dtype=torch.int32),
            halted=torch.zeros(batch_size, dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items() if v.numel() > 0}
        )

    def forward(self, carry: RF_TRM_Carry, batch: Dict[str, torch.Tensor], return_keys=None):
        # batch keys: "image", "agent_pos", "action" (for training)
        
        images = batch["image"]
        agent_pos = batch["agent_pos"]
        
        obs_x = self.get_obs_embedding(images, agent_pos)
        
        if self.training:
            actions = batch["action"] # (B, chunk_len, action_dim)
            loss = self.training_step(obs_x, actions)
            metrics = {"loss": loss.detach()}
            return carry, loss, metrics, {}, True
        else:
            y_pred = self.inference(obs_x)
            preds = {"action": y_pred}
            return carry, torch.tensor(0.0), {}, preds, True

    def training_step(self, obs_x, y_1):
        # Unrolled Flow Matching
        B = obs_x.shape[0]
        device = obs_x.device
        
        y_0 = torch.randn_like(y_1)
        z = self.initialize_latent(B, device)
        
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
        z = self.initialize_latent(B, device)
        
        dt = 1.0 / self.config.num_steps
        
        for step in range(self.config.num_steps):
            t = step / self.config.num_steps
            
            v_pred, z_next = self.net(obs_x, y, z, t)
            
            y = y + v_pred * dt
            z = z_next
            
        return y
