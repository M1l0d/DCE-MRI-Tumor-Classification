import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepyNetV2(nn.Module):
    def __init__(self, in_channels=2, tabular_dim=17, dropout=0.3):
        super(DeepyNetV2, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Output shape: [B, 128, 1, 1, 1]
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 + tabular_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x_img, x_tabular):
        x = self.conv_block(x_img)
        x = self.flatten(x)
        x = torch.cat([x, x_tabular], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
