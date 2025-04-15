import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv3DNet(nn.Module):
    def __init__(self, in_channels=4):
            super().__init__()
            self.features = nn.Sequential(
                # Input: 4 x 64 x 64 x 64
                nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.MaxPool3d(2),  # 16 x 16 x 16 x 16
                
                nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.MaxPool3d(2),  # 32 x 8 x 8 x 8
                
                nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1)  # 64 x 1 x 1 x 1
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, 1)
            )
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- CBAM Modules ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

    def forward(self, x):
        B, C, _, _, _ = x.size()
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        avg_att = self.mlp(avg_out)
        max_att = self.mlp(max_out)
        att = torch.sigmoid(avg_att + max_att).view(B, C, 1, 1, 1)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = torch.sigmoid(self.conv(att))
        return x * att


class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# --- Residual Blocks with GroupNorm ---
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class BottleneckBlock3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(8, planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(8, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# --- Backbone ResNet3D ---
class ResNet3D(nn.Module):
    def __init__(self, block, layers, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 64  # FIX: set after conv1, before layer1

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # CBAM (if used)
        self.cbam = CBAM3D(512 * block.expansion)

        # Output features
        self._out_features = 512 * block.expansion

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion),
            )

        layers = [block(self.in_channels, planes, stride, downsample)]
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        return x

# --- ResNet Factory ---
def resnet18_3d(in_channels=3):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels=in_channels)

def resnet50_3d(in_channels=3):
    return ResNet3D(BottleneckBlock3D, [3, 4, 6, 3], in_channels=in_channels)

# --- TNBCNet with Attention ---
class TNBCNet(nn.Module):
    def __init__(self, in_channels, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = resnet18_3d(in_channels=in_channels)
        elif backbone == "resnet50":
            self.backbone = resnet50_3d(in_channels=in_channels)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        feature_dim = self.backbone._out_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout for regularization
            nn.Linear(256, 1)
        )

    def forward(self, x_img, x_tabular=None):
        x = self.backbone.forward_features(x_img)  # (B, C, D, H, W)
        x = torch.mean(x, dim=(2, 3, 4))  # Global Average Pooling
        return self.classifier(x)