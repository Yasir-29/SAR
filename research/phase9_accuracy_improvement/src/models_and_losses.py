import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

# -------------------------------------------------------------
# Base ResNet34 Encoder
# -------------------------------------------------------------
class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=2):
        super(ResNet34Encoder, self).__init__()
        self.resnet = models.resnet34(weights=None)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

# -------------------------------------------------------------
# Architectural Variants (Experiment D & E)
# -------------------------------------------------------------
class BaselineTamilNaduTransferNet(nn.Module):
    """Certified Phase 9 Production Architecture: Dual ResNet34 + Absolute Diff"""
    def __init__(self, num_classes=3, dropout=0.2):
        super(BaselineTamilNaduTransferNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1536, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_pre, x_post):
        pre_feat_map = self.pre_encoder(x_pre)
        post_feat_map = self.post_encoder(x_post)
        change_feat_map = torch.abs(post_feat_map - pre_feat_map)

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        change_feat = self.pool(change_feat_map).view(x_pre.size(0), -1)

        fused = torch.cat([pre_feat, post_feat, change_feat], dim=1) # 1536
        x = F.relu(self.fc1(self.dropout(fused)))
        logits = self.fc2(x)
        return logits

class LayerNormTamilNaduNet(nn.Module):
    """LayerNorm Refined Classifier Head"""
    def __init__(self, num_classes=3, dropout=0.3):
        super(LayerNormTamilNaduNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ln = nn.LayerNorm(1536)
        self.fc1 = nn.Linear(1536, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_pre, x_post):
        pre_feat_map = self.pre_encoder(x_pre)
        post_feat_map = self.post_encoder(x_post)
        change_feat_map = torch.abs(post_feat_map - pre_feat_map)

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        change_feat = self.pool(change_feat_map).view(x_pre.size(0), -1)

        fused = torch.cat([pre_feat, post_feat, change_feat], dim=1) # 1536
        fused = self.ln(fused)
        x = F.relu(self.fc1(self.dropout(fused)))
        logits = self.fc2(x)
        return logits

class SignedDiffTamilNaduNet(nn.Module):
    """Signed Feature Difference Model: (POST - PRE)"""
    def __init__(self, num_classes=3, dropout=0.2):
        super(SignedDiffTamilNaduNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_pre, x_post):
        pre_feat_map = self.pre_encoder(x_pre)
        post_feat_map = self.post_encoder(x_post)
        signed_diff_map = post_feat_map - pre_feat_map

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        diff_feat = self.pool(signed_diff_map).view(x_pre.size(0), -1)

        fused = torch.cat([pre_feat, post_feat, diff_feat], dim=1) # 1536
        x = self.fc1(fused)
        logits = self.fc2(x)
        return logits

class ConcatTemporalTamilNaduNet(nn.Module):
    """Concatenated PRE, POST, Absolute Diff, and Signed Diff (2048-dim)"""
    def __init__(self, num_classes=3, dropout=0.2):
        super(ConcatTemporalTamilNaduNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_pre, x_post):
        pre_feat_map = self.pre_encoder(x_pre)
        post_feat_map = self.post_encoder(x_post)
        abs_diff_map = torch.abs(post_feat_map - pre_feat_map)
        signed_diff_map = post_feat_map - pre_feat_map

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        abs_diff_feat = self.pool(abs_diff_map).view(x_pre.size(0), -1)
        signed_diff_feat = self.pool(signed_diff_map).view(x_pre.size(0), -1)

        fused = torch.cat([pre_feat, post_feat, abs_diff_feat, signed_diff_feat], dim=1) # 2048
        x = self.fc1(fused)
        logits = self.fc2(x)
        return logits

class LogRatioTamilNaduNet(nn.Module):
    """Log-Ratio Feature Difference Model: log(|POST| + eps) - log(|PRE| + eps)"""
    def __init__(self, num_classes=3, dropout=0.2):
        super(LogRatioTamilNaduNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_pre, x_post):
        pre_feat_map = self.pre_encoder(x_pre)
        post_feat_map = self.post_encoder(x_post)
        
        eps = 1e-5
        log_ratio_map = torch.log(torch.abs(post_feat_map) + eps) - torch.log(torch.abs(pre_feat_map) + eps)

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        log_ratio_feat = self.pool(log_ratio_map).view(x_pre.size(0), -1)

        fused = torch.cat([pre_feat, post_feat, log_ratio_feat], dim=1)
        x = self.fc1(fused)
        logits = self.fc2(x)
        return logits

class AttentionFusionTamilNaduNet(nn.Module):
    """Cross-Temporal Gated Attention Fusion Network"""
    def __init__(self, num_classes=3, dropout=0.2):
        super(AttentionFusionTamilNaduNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)

        self.att_gate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(1536, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_pre, x_post):
        pre_feat_map = self.pre_encoder(x_pre)
        post_feat_map = self.post_encoder(x_post)
        diff_map = post_feat_map - pre_feat_map

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        diff_feat = self.pool(diff_map).view(x_pre.size(0), -1)

        concat_pre_post = torch.cat([pre_feat, post_feat], dim=1) # 1024
        att_weights = self.att_gate(concat_pre_post) # 512
        gated_diff = diff_feat * att_weights

        fused = torch.cat([pre_feat, post_feat, gated_diff], dim=1) # 1536
        x = self.fc1(fused)
        logits = self.fc2(x)
        return logits

# -------------------------------------------------------------
# Loss Functions (Experiment A)
# -------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ClassBalancedLoss(nn.Module):
    """Effective Number of Samples Class Balanced Loss (Cui et al., CVPR 2019)"""
    def __init__(self, samples_per_cls=[4362, 654, 171], beta=0.999, loss_type='cross_entropy', gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.loss_type = loss_type
        self.gamma = gamma

    def forward(self, inputs, targets):
        weights = self.weights.to(inputs.device)
        if self.loss_type == 'cross_entropy':
            return F.cross_entropy(inputs, targets, weight=weights)
        elif self.loss_type == 'focal':
            focal = FocalLoss(alpha=weights, gamma=self.gamma)
            return focal(inputs, targets)
        return F.cross_entropy(inputs, targets, weight=weights)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)
        smooth_target = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -torch.sum(smooth_target * log_probs, dim=-1)
        if self.weight is not None:
            weight_t = self.weight.to(inputs.device)[targets]
            loss = loss * weight_t
        return loss.mean()

# -------------------------------------------------------------
# Temperature Scaler for Calibration (Experiment H)
# -------------------------------------------------------------
class TemperatureScaler(nn.Module):
    """Post-hoc Temperature Scaling Calibration (Guo et al., ICML 2017)"""
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, val_logits, val_labels, lr=0.01, max_iter=50):
        val_logits = torch.tensor(val_logits, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.long)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature.item()
