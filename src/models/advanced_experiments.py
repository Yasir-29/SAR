import os
import sys
import json
import random
import copy
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.train_and_evaluate import (
    set_seed,
    TamilNaduChipDataset,
    ResNet34Encoder,
    BaselineTamilNaduTransferNet,
    AttentionFusionTransferNet,
    FocalLoss,
    evaluate_model,
    calculate_ece
)

# -------------------------------------------------------------
# 1. Advanced Architecture Definitions
# -------------------------------------------------------------

class MultiScaleTamilNaduNet(nn.Module):
    """Multi-Scale Architecture combining local crop (32x32) and downsampled context"""
    def __init__(self, num_classes=3):
        super(MultiScaleTamilNaduNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Separate MLP for multi-scale fusion
        self.fc1 = nn.Linear(1536 + 512, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_pre, x_post):
        # Local scale
        pre_feat_map = self.pre_encoder(x_pre)
        post_feat_map = self.post_encoder(x_post)
        change_feat_map = torch.abs(post_feat_map - pre_feat_map)

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        change_feat = self.pool(change_feat_map).view(x_pre.size(0), -1)

        # Context scale: downsampled by 2
        x_pre_down = F.interpolate(x_pre, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_post_down = F.interpolate(x_post, scale_factor=0.5, mode='bilinear', align_corners=False)
        ctx_map = torch.abs(self.post_encoder(x_post_down) - self.pre_encoder(x_pre_down))
        ctx_feat = self.pool(ctx_map).view(x_pre.size(0), -1)

        fused = torch.cat([pre_feat, post_feat, change_feat, ctx_feat], dim=1)
        x = F.relu(self.fc1(self.dropout(fused)))
        return self.fc2(x)


class HierarchicalTamilNaduNet(nn.Module):
    """2-Stage Hierarchical Classifier Network:
       Stage 1: INTACT vs NON-INTACT (Damaged or Destroyed)
       Stage 2: DAMAGED vs DESTROYED
    """
    def __init__(self):
        super(HierarchicalTamilNaduNet, self).__init__()
        self.stage1_net = BaselineTamilNaduTransferNet(num_classes=2)
        self.stage2_net = BaselineTamilNaduTransferNet(num_classes=2)

    def forward(self, x_pre, x_post):
        logits_stage1 = self.stage1_net(x_pre, x_post) # [B, 2] -> (0: Intact, 1: Damaged/Destroyed)
        logits_stage2 = self.stage2_net(x_pre, x_post) # [B, 2] -> (0: Damaged, 1: Destroyed)

        p_stage1 = F.softmax(logits_stage1, dim=1)
        p_stage2 = F.softmax(logits_stage2, dim=1)

        p_intact = p_stage1[:, 0:1]
        p_non_intact = p_stage1[:, 1:2]

        p_damaged = p_non_intact * p_stage2[:, 0:1]
        p_destroyed = p_non_intact * p_stage2[:, 1:2]

        probs_3class = torch.cat([p_intact, p_damaged, p_destroyed], dim=1)
        # Convert back to pseudo-logits via log for standard loss/eval compatibility
        logits_3class = torch.log(probs_3class + 1e-7)
        return logits_3class


class SignedDiffTamilNaduNet(nn.Module):
    """Architecture utilizing both Signed (POST - PRE) and Absolute Differences"""
    def __init__(self, num_classes=3):
        super(SignedDiffTamilNaduNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_pre, x_post):
        pre_feat_map = self.pre_encoder(x_pre)
        post_feat_map = self.post_encoder(x_post)

        abs_diff_map = torch.abs(post_feat_map - pre_feat_map)
        signed_diff_map = post_feat_map - pre_feat_map

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        abs_diff_feat = self.pool(abs_diff_map).view(x_pre.size(0), -1)
        signed_diff_feat = self.pool(signed_diff_map).view(x_pre.size(0), -1)

        fused = torch.cat([pre_feat, post_feat, abs_diff_feat, signed_diff_feat], dim=1)
        x = F.relu(self.fc1(self.dropout(fused)))
        return self.fc2(x)

# -------------------------------------------------------------
# 2. Multi-Seed Training Helper
# -------------------------------------------------------------

def train_model_single_seed(model_factory, train_dataset, val_dataset, loss_fn, seed=42, epochs=15, batch_size=32, lr=1e-4, head_lr=1e-3, device='cpu'):
    set_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model_factory().to(device)

    # Differential learning rates
    if hasattr(model, 'fc1') and hasattr(model, 'pre_encoder'):
        optimizer = optim.AdamW([
            {'params': model.pre_encoder.parameters(), 'lr': lr},
            {'params': model.post_encoder.parameters(), 'lr': lr},
            {'params': model.fc1.parameters(), 'lr': head_lr},
            {'params': model.fc2.parameters(), 'lr': head_lr},
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_macro_f1 = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for pre_t, post_t, targets in train_loader:
            pre_t, post_t, targets = pre_t.to(device), post_t.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(pre_t, post_t)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        val_macro_f1 = val_metrics["macro_f1"]

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_val_metrics, _, _ = evaluate_model(model, val_loader, device)
    return model, final_val_metrics
