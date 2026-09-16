import os
import sys
import json
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import torchvision.models as models

# Set fixed random seeds for complete reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------
# Datasets & Augmentations
# -------------------------------------------------------------
class TamilNaduChipDataset(Dataset):
    def __init__(self, json_file, augment=False, label_shuffle=False, pre_only=False, post_only=False, change_only=False, zero_post=False, shuffle_post=False):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.augment = augment
        self.pre_only = pre_only
        self.post_only = post_only
        self.change_only = change_only
        self.zero_post = zero_post
        self.shuffle_post = shuffle_post

        self.labels = [item["verified_label"] for item in self.data]
        if label_shuffle:
            # Shuffle labels deterministically for control test
            shuffled_labels = list(self.labels)
            random.Random(42).shuffle(shuffled_labels)
            self.labels = shuffled_labels

        # Pre-index post chips if shuffling post modality
        if self.shuffle_post:
            self.all_post_paths = [item["sar_post_path"] for item in self.data]
            random.Random(42).shuffle(self.all_post_paths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pre_path = item["sar_pre_path"]
        post_path = self.all_post_paths[idx] if self.shuffle_post else item["sar_post_path"]
        label = self.labels[idx]

        pre = np.load(pre_path).astype(np.float32)  # (2, 32, 32)
        post = np.load(post_path).astype(np.float32)  # (2, 32, 32)

        if self.zero_post:
            post = np.zeros_like(post)

        if self.pre_only:
            post = np.copy(pre)
        elif self.post_only:
            pre = np.copy(post)

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                pre = np.flip(pre, axis=2).copy()
                post = np.flip(post, axis=2).copy()
            # Random vertical flip
            if random.random() > 0.5:
                pre = np.flip(pre, axis=1).copy()
                post = np.flip(post, axis=1).copy()
            # Random 90 deg rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                pre = np.rot90(pre, k, axes=(1, 2)).copy()
                post = np.rot90(post, k, axes=(1, 2)).copy()
            # Mild SAR noise jitter
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.02, size=pre.shape).astype(np.float32)
                pre = pre + noise
                post = post + noise

        pre_t = torch.from_numpy(pre)
        post_t = torch.from_numpy(post)
        if self.change_only:
            pre_t = torch.abs(post_t - pre_t)
            post_t = torch.abs(post_t - pre_t)

        return pre_t, post_t, torch.tensor(label, dtype=torch.long)

# -------------------------------------------------------------
# Loss Functions
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

# -------------------------------------------------------------
# Model Architectures
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

class BaselineTamilNaduTransferNet(nn.Module):
    """Certified Phase 9 Baseline Production Model"""
    def __init__(self, num_classes=3):
        super(BaselineTamilNaduTransferNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1536, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_pre, x_post):
        pre_feat_map = self.pre_encoder(x_pre)
        post_feat_map = self.post_encoder(x_post)
        change_feat_map = torch.abs(post_feat_map - pre_feat_map)

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        change_feat = self.pool(change_feat_map).view(x_pre.size(0), -1)

        fused = torch.cat([pre_feat, post_feat, change_feat], dim=1)
        x = F.relu(self.fc1(self.dropout(fused)))
        logits = self.fc2(x)
        return logits

class AttentionFusionTransferNet(nn.Module):
    """Advanced Candidate Model with Cross-Temporal Channel & Spatial Attention"""
    def __init__(self, num_classes=3):
        super(AttentionFusionTransferNet, self).__init__()
        self.pre_encoder = ResNet34Encoder(in_channels=2)
        self.post_encoder = ResNet34Encoder(in_channels=2)

        # Cross-temporal attention gating
        self.att_fc = nn.Sequential(
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
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_pre, x_post):
        pre_feat_map = self.pre_encoder(x_pre)   # (B, 512, 1, 1)
        post_feat_map = self.post_encoder(x_post) # (B, 512, 1, 1)
        diff_map = post_feat_map - pre_feat_map  # Signed difference

        pre_feat = self.pool(pre_feat_map).view(x_pre.size(0), -1)
        post_feat = self.pool(post_feat_map).view(x_post.size(0), -1)
        diff_feat = self.pool(diff_map).view(x_pre.size(0), -1)

        concat_pre_post = torch.cat([pre_feat, post_feat], dim=1)
        att_weights = self.att_fc(concat_pre_post)
        gated_diff = diff_feat * att_weights

        fused = torch.cat([pre_feat, post_feat, gated_diff], dim=1)
        x = self.fc1(fused)
        logits = self.fc2(x)
        return logits

# -------------------------------------------------------------
# Calibration (ECE) Computation
# -------------------------------------------------------------
def calculate_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)

# -------------------------------------------------------------
# Evaluation Helper
# -------------------------------------------------------------
def evaluate_model(model, dataloader, device, thresholds=None):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for pre_t, post_t, targets in dataloader:
            pre_t, post_t = pre_t.to(device), post_t.to(device)
            logits = model(pre_t, post_t)
            all_logits.append(logits.cpu())
            all_targets.append(targets)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    probs = F.softmax(all_logits, dim=1).numpy()

    if thresholds is not None:
        # Apply custom probability threshold adjustments
        adjusted_probs = probs * np.array(thresholds)
        preds = np.argmax(adjusted_probs, axis=1)
    else:
        preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(all_targets, preds))
    macro_f1 = float(f1_score(all_targets, preds, average='macro'))
    weighted_f1 = float(f1_score(all_targets, preds, average='weighted'))
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, preds, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(all_targets, preds, labels=[0, 1, 2])
    balanced_acc = float(np.mean(rec))
    ece = calculate_ece(probs, all_targets)

    metrics = {
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(balanced_acc * 100, 2),
        "precision_per_class": [round(p, 4) for p in prec],
        "recall_per_class": [round(r, 4) for r in rec],
        "f1_per_class": [round(f, 4) for f in f1],
        "intact_f1": round(f1[0], 4),
        "damaged_f1": round(f1[1], 4),
        "destroyed_f1": round(f1[2], 4),
        "intact_recall": round(rec[0] * 100, 2),
        "damaged_recall": round(rec[1] * 100, 2),
        "destroyed_recall": round(rec[2] * 100, 2),
        "confusion_matrix": cm.tolist(),
        "calibration_ece": round(ece, 4)
    }
    return metrics, probs, all_targets

# -------------------------------------------------------------
# Validation Threshold Tuning (Validation Set ONLY)
# -------------------------------------------------------------
def tune_validation_thresholds(probs, targets):
    """Grid search over validation set probabilities to maximize validation Macro-F1."""
    best_macro_f1 = 0.0
    best_weights = [1.0, 1.0, 1.0]

    for w1 in np.linspace(0.5, 3.0, 11):
        for w2 in np.linspace(0.5, 5.0, 19):
            weights = [1.0, w1, w2]
            adj_probs = probs * np.array(weights)
            preds = np.argmax(adj_probs, axis=1)
            f1 = f1_score(targets, preds, average='macro')
            if f1 > best_macro_f1:
                best_macro_f1 = f1
                best_weights = weights

    return best_weights, best_macro_f1

if __name__ == "__main__":
    print("Training and Evaluation harness loaded successfully.")
