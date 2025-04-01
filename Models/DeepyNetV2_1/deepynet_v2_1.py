# deepynet_v2_1.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class DeepyNetV2_1(nn.Module):
    def __init__(self, in_channels=2, tabular_dim=17, dropout=0.3):
        super(DeepyNetV2_1, self).__init__()
        act_fn = Swish()

        # -------- CNN Backbone --------
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(),
            nn.MaxPool3d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),
            nn.MaxPool3d(2)
        )

        # Global Pooling & Projection
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.img_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            act_fn,
            nn.Dropout(dropout)
        )

        # -------- MLP Fusion Head --------
        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + tabular_dim, 128),
            nn.BatchNorm1d(128),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_tabular):
        x = self.conv_block1(x_img)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = self.img_fc(x)

        # Fuse with tabular data
        x = torch.cat([x, x_tabular], dim=1)
        x = self.fusion_fc(x)
        return x
