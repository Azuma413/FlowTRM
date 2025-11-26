from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import CastedLinear
from models.recursive_reasoning.rf_trm import RF_TRM_Net, RF_TRM_Config
from models.vision import ResNet18Encoder

class PushT_RF_TRM_Config(RF_TRM_Config):
    # PushT specific defaults
    action_dim: int = 2 # PushT has 2D action (x, y)
    prop_dim: int = 2 # Agent pos (x, y)
    image_size: int = 96
    
    # Legacy fields (required by base class but unused)
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Training forward pass.
        Returns: loss, metrics
        """
        images = batch["image"]
        agent_pos = batch["agent_pos"]
        actions = batch["action"] # (B, chunk_len, action_dim)
        
        obs_x = self.get_obs_embedding(images, agent_pos)
        loss = self.training_step(obs_x, actions)
        
        return loss, {"loss": loss.detach()}

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Inference forward pass.
        Returns: predictions dict
        """
        images = batch["image"]
        agent_pos = batch["agent_pos"]
        
        obs_x = self.get_obs_embedding(images, agent_pos)
        y_pred = self.inference(obs_x)
        
        return {"action": y_pred}

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
