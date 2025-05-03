import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class TNBCNet(nn.Module):
    def __init__(self, in_channels):
        super(TNBCNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.block1 = BasicBlock3D(32, 32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.block2 = BasicBlock3D(64, 64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.block3 = BasicBlock3D(128, 128)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.block2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
