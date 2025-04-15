# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class TNBCNet(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm3d(32)
#         self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(64)
#         self.pool = nn.MaxPool3d(2)

#         self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm3d(128)

#         self.global_pool = nn.AdaptiveAvgPool3d(1)
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x_img, x_tabular=None):
#         x = self.pool(F.relu(self.bn1(self.conv1(x_img))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.global_pool(x)
#         out = self.classifier(x)
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class TNBCNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm3d(64)
        self.pool = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.in3 = nn.InstanceNorm3d(128)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_img, x_tabular=None):
        x = self.pool(F.relu(self.in1(self.conv1(x_img))))
        x = self.pool(F.relu(self.in2(self.conv2(x))))
        x = self.pool(F.relu(self.in3(self.conv3(x))))
        x = self.global_pool(x)
        out = self.classifier(x)
        return out
