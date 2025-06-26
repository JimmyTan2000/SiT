import torch
import torch.nn as nn

class PoseEncoder(nn.Module):
    def __init__(self, latent_size=32, channels=4, hidden_dim=256):
        super().__init__()
        self.latent_size = latent_size
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(16, hidden_dim),  # 16 for flattened 4x4 pose
            nn.ReLU(),
            nn.Linear(hidden_dim, channels * latent_size * latent_size)
        )

    def forward(self, pose):  # pose: (B, 4, 4) or (B, 16)
        if pose.dim() == 3:  # (B, 4, 4)
            pose = pose.view(pose.size(0), -1)  # (B, 16)
        x = self.fc(pose)  # (B, channels * latent_size * latent_size)
        x = x.view(-1, self.channels, self.latent_size, self.latent_size)
        return x
