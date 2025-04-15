import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepyNetV3(nn.Module):
    def __init__(self, in_channels=2, tabular_dim=17, dropout=0.3):
        super(DeepyNetV3, self).__init__()

        # 3D CNN Branch (Image)
        self.img_conv = nn.Sequential(
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
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Output: (B, 256, 1, 1, 1)
        )

        # MLP Branch (Tabular)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusion + Classification
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1)  # BCEWithLogitsLoss expects raw logits
        )

    def forward(self, x_img, x_tabular):
        x_img = self.img_conv(x_img)  # shape: (B, 256, 1, 1, 1)
        x_img = x_img.view(x_img.size(0), -1)  # Flatten to (B, 256)

        x_tab = self.tabular_mlp(x_tabular)  # (B, 64)

        x = torch.cat([x_img, x_tab], dim=1)  # (B, 320)
        out = self.classifier(x)  # (B, 1)

        return out
