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
    n_obs_steps: int = 2 # Number of observation steps to condition on

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
        
        # Projection for history
        # We concatenate features from n_obs_steps: (B, n_obs_steps * obs_dim) -> (B, obs_dim)
        self.history_projection = CastedLinear(
            self.config.n_obs_steps * self.config.obs_dim, 
            self.config.obs_dim, 
            bias=True
        )
        
        # Core Network (Delegated to Inner)
        from models.recursive_reasoning.rf_trm import RF_TRM_Inner
        self.inner = RF_TRM_Inner(self.config)
        
    def get_obs_embedding(self, images, agent_pos):
        # images: (B, T, C, H, W)
        # agent_pos: (B, T, prop_dim)
        
        B, T, C, H, W = images.shape
        
        # Flatten B and T for encoder
        images_flat = images.view(B * T, C, H, W)
        agent_pos_flat = agent_pos.view(B * T, -1)
        
        img_emb = self.vision_encoder(images_flat) # (B*T, obs_dim//2)
        prop_emb = self.prop_encoder(agent_pos_flat) # (B*T, obs_dim//2)
        
        # Concatenate
        obs_flat = torch.cat([img_emb, prop_emb], dim=-1) # (B*T, obs_dim)
        
        # Reshape back to (B, T, obs_dim)
        obs_seq = obs_flat.view(B, T, -1)
        
        # Flatten T into feature dim -> (B, T * obs_dim)
        obs_cat = obs_seq.view(B, -1)
        
        # Project back to obs_dim
        obs_x = self.history_projection(obs_cat) # (B, obs_dim)
        
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
        
        # Delegate to Inner
        loss = self.inner.training_step(obs_x, actions)
        
        return loss, {"loss": loss.detach()}

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Inference forward pass.
        Returns: predictions dict
        """
        images = batch["image"]
        agent_pos = batch["agent_pos"]
        
        obs_x = self.get_obs_embedding(images, agent_pos)
        
        # Delegate to Inner
        y_pred = self.inner.inference(obs_x)
        
        return {"action": y_pred}
