import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Self-Attention Module ---
class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv3d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        
        # Reshape for attention
        query = self.query(x).view(batch_size, -1, D*H*W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, D*H*W)
        value = self.value(x).view(batch_size, -1, D*H*W)
        
        # Attention map
        attention = F.softmax(torch.bmm(query, key), dim=2)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)
        
        # Residual connection
        out = self.gamma * out + x
        return out

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

# --- Residual Blocks with BatchNorm ---
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.cbam = CBAM3D(planes)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention
        out = self.cbam(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return self.relu(out)

class BottleneckBlock3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.cbam = CBAM3D(planes * self.expansion)

    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Apply attention
        out = self.cbam(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return self.relu(out)

# --- Backbone ResNet3D ---
class ResNet3D(nn.Module):
    def __init__(self, block, layers, in_channels=3, dropout_rate=0.5):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
            
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        
        self.in_planes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
            
        return nn.Sequential(*layers)
        
    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
        
    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        return x.view(x.size(0), -1)

# --- ResNet Factory ---
def resnet18_3d(in_channels=3):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels=in_channels)

def resnet50_3d(in_channels=3):
    return ResNet3D(BottleneckBlock3D, [3, 4, 6, 3], in_channels=in_channels)

# --- TNBCNet with Attention ---
class TNBCNet(nn.Module):
    def __init__(self, in_channels, backbone="resnet18", dropout_rate=0.5):
        super().__init__()
        
        # Choose backbone
        if backbone == "resnet18":
            self.backbone = resnet18_3d(in_channels)
            self.feature_dim = 512
        elif backbone == "resnet50":
            self.backbone = resnet50_3d(in_channels)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1, bias=True)
        )
        
        # Initialize weights
        self._initialize_weights()
        with torch.no_grad():
            self.classifier[-1].bias.fill_(torch.log(torch.tensor(1/3)))  # log(p/(1-p)) where p is positive class frequency
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, x_tabular=None):
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        return logits