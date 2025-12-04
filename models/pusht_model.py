import torch
import torch.nn as nn
import torchvision.models as models
from models.irm import IterativeRefinementModel

class PushTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone (ResNet18)
        # Assuming 96x96 images, ResNet18 is reasonable.
        # We replace the final fc layer with Identity or remove it.
        resnet = models.resnet18(weights=None) # Train from scratch or load weights? Usually scratch for simple tasks or pretrained.
        # For PushT, pretrained is better but let's stick to scratch if not specified, 
        # or maybe pretrained=True if we can access internet. User has internet.
        # But let's use weights='IMAGENET1K_V1' if possible.
        # However, to be safe and avoid download issues if not requested, I'll use None or check config.
        # The original code didn't show backbone init. I'll use default (random init).
        
        # Remove fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone_dim = 512
        
        # IRM
        # obs_dim = backbone_dim + state_dim * n_obs_steps
        # Wait, PushT usually stacks multiple frames?
        # dataset returns (n_obs_steps, C, H, W).
        # We need to encode each frame or stack channel-wise?
        # Usually stack channel-wise: (C * n_obs_steps, H, W) -> ResNet -> Vector.
        # Or ResNet on each frame -> RNN/Transformer.
        # IRM is a transformer/MLP based.
        # Config says `n_obs_steps: int = 2`.
        # `PushTDataset` returns `image` as (n_obs_steps, C, H, W).
        # We should probably flatten n_obs_steps into channels: (n_obs_steps * C, H, W).
        # ResNet18 expects 3 channels.
        # If we have 6 channels, we need to modify the first conv.
        
        self.n_obs_steps = config.get("n_obs_steps", 2)
        self.input_channels = 3 * self.n_obs_steps
        
        # Modify first conv to accept input_channels
        self.backbone[0] = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.state_dim = config.get("prop_dim", 2) * self.n_obs_steps
        
        self.irm_obs_dim = self.backbone_dim + self.state_dim
        
        self.irm = IterativeRefinementModel(
            action_dim=config["action_dim"],
            obs_dim=self.irm_obs_dim,
            hidden_dim=config["hidden_size"],
            chunk_size=config["chunk_len"],
            train_noise_std=0.1 # Default
        )
        
    def forward(self, batch):
        # batch['image']: (B, T, C, H, W)
        # batch['agent_pos']: (B, T, D)
        # batch['action']: (B, chunk_len, action_dim)
        
        imgs = batch['image']
        states = batch['agent_pos']
        
        B, T, C, H, W = imgs.shape
        # Flatten T into C
        imgs = imgs.view(B, T*C, H, W)
        states = states.view(B, T*states.shape[-1])
        
        img_emb = self.backbone(imgs).flatten(1) # (B, 512)
        
        obs = torch.cat([img_emb, states], dim=-1) # (B, 512 + state_dim)
        
        irm_batch = {
            'obs': obs,
            'action': batch['action']
        }
        
        return self.irm.compute_loss(irm_batch), {}

    def predict(self, batch, prev_action_chunk=None):
        # batch['image']: (B, T, C, H, W)
        # batch['agent_pos']: (B, T, D)
        
        imgs = batch['image']
        states = batch['agent_pos']
        
        B, T, C, H, W = imgs.shape
        imgs = imgs.view(B, T*C, H, W)
        states = states.view(B, T*states.shape[-1])
        
        img_emb = self.backbone(imgs).flatten(1)
        obs = torch.cat([img_emb, states], dim=-1)
        
        action = self.irm.predict_with_warm_start(obs, prev_action_chunk=prev_action_chunk)
        
        return {"action": action}
