import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGuidedPooling(nn.Module):
    def __init__(self):
        super(MaskGuidedPooling, self).__init__()

    def forward(self, x, mask):
        # x shape: (batch_size, channels, H, W)
        # mask shape: (batch_size, 1, 32, 32)
        batch_size, channels, H, W = x.size()
        
        # Resize mask to (H, W)
        mask_down = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False) # (batch_size, 1, H, W)
        
        # Calculate mask-guided pooling
        # Prevent division by zero with small epsilon
        denom = torch.sum(mask_down, dim=(2, 3), keepdim=True) + 1e-5
        pooled = torch.sum(x * mask_down, dim=(2, 3), keepdim=True) / denom # (batch_size, channels, 1, 1)
        
        return pooled.view(batch_size, channels)

class LightweightConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super(LightweightConvEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Input shape: (batch_size, in_channels, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 16x16
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x)))) # 8x8
        return x

class BuildingDamageClassifier(nn.Module):
    def __init__(self, use_mask=True, use_optical=False):
        super(BuildingDamageClassifier, self).__init__()
        self.use_mask = use_mask
        self.use_optical = use_optical
        
        # Encoders
        self.sar_encoder = LightweightConvEncoder(in_channels=6)
        
        if use_optical:
            self.opt_encoder = LightweightConvEncoder(in_channels=8)
            fusion_dim = 256 + 256
        else:
            fusion_dim = 256
            
        # Pooling
        if use_mask:
            self.pool = MaskGuidedPooling()
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            
        # MLP Head
        self.fc1 = nn.Linear(fusion_dim, 64)
        self.fc2 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, s1, mask, s2=None, opt_available=None):
        # SAR Encoders
        sar_feat_map = self.sar_encoder(s1) # (batch_size, 256, 8, 8)
        
        if self.use_mask:
            sar_feat = self.pool(sar_feat_map, mask) # (batch_size, 256)
        else:
            sar_feat = self.pool(sar_feat_map).view(s1.size(0), -1)
            
        if self.use_optical:
            opt_feat_map = self.opt_encoder(s2)
            if self.use_mask:
                opt_feat = self.pool(opt_feat_map, mask)
            else:
                opt_feat = self.pool(opt_feat_map).view(s2.size(0), -1)
                
            # Apply optical mask/availability gate
            if opt_available is not None:
                gate = opt_available.view(-1, 1)
                opt_feat = opt_feat * gate
                
            # Feature Fusion
            feat = torch.cat([sar_feat, opt_feat], dim=1) # (batch_size, 512)
        else:
            feat = sar_feat # (batch_size, 256)
            
        # Classifier MLP Head
        x = F.relu(self.fc1(self.dropout(feat)))
        logits = self.fc2(x) # (batch_size, 3)
        
        return logits

# ============================================================
# PHASE 5.6 NEW RESIDUAL CNN & DUAL-STREAM ARCHITECTURES
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class StrongSingleStreamCNN(nn.Module):
    def __init__(self, in_channels=7, num_classes=3):
        super(StrongSingleStreamCNN, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        
        self.layer1 = ResidualBlock(64, 64, stride=1)      # 32x32
        self.pool1 = nn.MaxPool2d(2)                       # 16x16
        
        self.layer2 = ResidualBlock(64, 128, stride=1)     # 16x16
        self.pool2 = nn.MaxPool2d(2)                       # 8x8
        
        self.layer3 = ResidualBlock(128, 256, stride=1)    # 8x8
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        # mask is ignored for single-stream CNNs
        out = F.relu(self.bn(self.conv(x)))
        out = self.pool1(self.layer1(out))
        out = self.pool2(self.layer2(out))
        out = self.layer3(out)
        out = self.pool(out).view(out.size(0), -1)
        out = F.relu(self.fc1(self.dropout(out)))
        out = self.fc2(out)
        return out

class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=128):
        super(SimpleCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 8x8
        return x

class MultiStreamTemporalNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiStreamTemporalNet, self).__init__()
        
        self.pre_encoder = SimpleCNNEncoder(in_channels=2, out_channels=128)
        self.post_encoder = SimpleCNNEncoder(in_channels=2, out_channels=128)
        self.change_encoder = SimpleCNNEncoder(in_channels=2, out_channels=128)
        
        self.pool = MaskGuidedPooling()
        
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_pre, x_post, x_change, mask):
        pre_feats = self.pre_encoder(x_pre)
        post_feats = self.post_encoder(x_post)
        change_feats = self.change_encoder(x_change)
        
        pre_pooled = self.pool(pre_feats, mask)
        post_pooled = self.pool(post_feats, mask)
        change_pooled = self.pool(change_feats, mask)
        
        fused = torch.cat([pre_pooled, post_pooled, change_pooled], dim=1) # (batch_size, 384)
        
        out = F.relu(self.fc1(self.dropout(fused)))
        out = self.fc2(out)
        return out
