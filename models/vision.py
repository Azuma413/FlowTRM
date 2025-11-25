import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True, freeze=False, output_dim=512):
        super().__init__()
        # Load ResNet18
        self.net = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        # ResNet18 structure: ... -> avgpool -> fc
        # We want the output of avgpool which is (B, 512, 1, 1) -> flatten -> (B, 512)
        self.net.fc = nn.Identity()
        
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False
                
        self.output_dim = 512 # ResNet18 default feature dim
        
        # Optional projection if output_dim is different
        self.projection = None
        if output_dim != 512:
            self.projection = nn.Linear(512, output_dim)
            self.output_dim = output_dim

    def forward(self, x):
        # x: (B, C, H, W)
        # Expects normalized images
        
        features = self.net(x) # (B, 512)
        
        if self.projection is not None:
            features = self.projection(features)
            
        return features
