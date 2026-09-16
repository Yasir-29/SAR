import os
import sys
import json
import random
import time
import copy
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.phase9_accuracy_improvement.src.models_and_losses import (
    BaselineTamilNaduTransferNet,
    LayerNormTamilNaduNet,
    SignedDiffTamilNaduNet,
    ConcatTemporalTamilNaduNet,
    LogRatioTamilNaduNet,
    AttentionFusionTamilNaduNet,
    FocalLoss,
    ClassBalancedLoss,
    LabelSmoothingCrossEntropy,
    TemperatureScaler
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------
# Dataset & Augmentation Pipeline (Experiment C)
# -------------------------------------------------------------
class TamilNaduChipDataset(Dataset):
    def __init__(self, json_file, augment=False, noise_std=0.0, shift_pixels=0):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.augment = augment
        self.noise_std = noise_std
        self.shift_pixels = shift_pixels
        self.labels = [item["verified_label"] for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pre = np.load(item["sar_pre_path"]).astype(np.float32)  # (2, 32, 32)
        post = np.load(item["sar_post_path"]).astype(np.float32)  # (2, 32, 32)
        label = self.labels[idx]

        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                pre = np.flip(pre, axis=2).copy()
                post = np.flip(post, axis=2).copy()
            # Vertical flip
            if random.random() > 0.5:
                pre = np.flip(pre, axis=1).copy()
                post = np.flip(post, axis=1).copy()
            # 90, 180, 270 rotations
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                pre = np.rot90(pre, k, axes=(1, 2)).copy()
                post = np.rot90(post, k, axes=(1, 2)).copy()
            # Speckle / Noise perturbation
            if self.noise_std > 0 and random.random() > 0.5:
                noise = np.random.normal(0, self.noise_std, size=pre.shape).astype(np.float32)
                pre += noise
                post += noise
            # Spatial Shift
            if self.shift_pixels > 0 and random.random() > 0.5:
                dx = random.randint(-self.shift_pixels, self.shift_pixels)
                dy = random.randint(-self.shift_pixels, self.shift_pixels)
                pre = np.roll(pre, shift=(dy, dx), axis=(1, 2))
                post = np.roll(post, shift=(dy, dx), axis=(1, 2))

        return torch.from_numpy(pre), torch.from_numpy(post), torch.tensor(label, dtype=torch.long)

# -------------------------------------------------------------
# Metric & Calibration Computation
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

def evaluate_loader(model, dataloader, device, thresholds=None, temperature_scaler=None):
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

    if temperature_scaler is not None:
        with torch.no_grad():
            scaled_logits = temperature_scaler(all_logits)
            probs = F.softmax(scaled_logits, dim=1).detach().numpy()
    else:
        probs = F.softmax(all_logits, dim=1).numpy()

    if thresholds is not None:
        adj_probs = probs * np.array(thresholds)
        preds = np.argmax(adj_probs, axis=1)
    else:
        preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(all_targets, preds))
    macro_f1 = float(f1_score(all_targets, preds, average='macro'))
    weighted_f1 = float(f1_score(all_targets, preds, average='weighted'))
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, preds, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(all_targets, preds, labels=[0, 1, 2])
    balanced_acc = float(np.mean(rec))
    ece = calculate_ece(probs, all_targets)

    # 95% Confidence Interval for Accuracy
    n = len(all_targets)
    ci95 = 1.96 * np.sqrt((acc * (1.0 - acc)) / n)
    ci95_str = f"{(acc - ci95)*100:.1f}% - {(acc + ci95)*100:.1f}%"

    metrics = {
        "accuracy": round(acc * 100, 2),
        "ci_95": ci95_str,
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
    return metrics, probs, all_logits.numpy(), all_targets

def save_confusion_matrix_plot(cm, title, filepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ['INTACT', 'DAMAGED', 'DESTROYED']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

# -------------------------------------------------------------
# Training Function
# -------------------------------------------------------------
def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler=None, epochs=15, device='cpu', early_stopping_patience=7):
    best_val_macro_f1 = 0.0
    best_state = None
    patience_counter = 0

    model = model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for pre_t, post_t, targets in train_loader:
            pre_t, post_t, targets = pre_t.to(device), post_t.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(pre_t, post_t)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(targets)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                val_m, _, _, _ = evaluate_loader(model, val_loader, device)
                scheduler.step(val_m["macro_f1"])
            else:
                scheduler.step()

        val_metrics, _, _, _ = evaluate_loader(model, val_loader, device)
        val_macro_f1 = val_metrics["macro_f1"]

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, val_probs, val_logits, val_targets = evaluate_loader(model, val_loader, device)
    return model, val_metrics, val_probs, val_logits, val_targets

# -------------------------------------------------------------
# Validation Threshold Search (Experiment G)
# -------------------------------------------------------------
def tune_validation_thresholds_constrained(probs, targets, min_damaged_rec=20.0, min_destroyed_rec=15.0):
    best_weights = [1.0, 1.0, 1.0]
    best_macro_f1 = 0.0

    for w1 in np.linspace(0.5, 3.0, 11):
        for w2 in np.linspace(0.5, 4.0, 15):
            weights = [1.0, w1, w2]
            adj_probs = probs * np.array(weights)
            preds = np.argmax(adj_probs, axis=1)

            prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average=None, labels=[0, 1, 2], zero_division=0)
            macro_f1 = f1_score(targets, preds, average='macro')
            d_rec, dest_rec = rec[1] * 100, rec[2] * 100

            if d_rec >= min_damaged_rec and dest_rec >= min_destroyed_rec:
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_weights = weights

    return best_weights, best_macro_f1

# -------------------------------------------------------------
# Main Experiment Execution Suite
# -------------------------------------------------------------
def run_full_experiment_suite():
    print("=" * 70)
    print("STARTING FULL PHASE 9 ACCURACY IMPROVEMENT RESEARCH SUITE")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Compute Device: {device}")

    base_dir = "research/phase9_accuracy_improvement"
    os.makedirs(f"{base_dir}/configs", exist_ok=True)
    os.makedirs(f"{base_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{base_dir}/results", exist_ok=True)
    os.makedirs(f"{base_dir}/figures", exist_ok=True)
    os.makedirs(f"{base_dir}/reports", exist_ok=True)

    train_json = "data/tamil_nadu/final/FINAL_TRAIN.json"
    val_json = "data/tamil_nadu/final/FINAL_VALIDATION.json"
    test_json = "data/tamil_nadu/final/FINAL_TEST.json"
    holdout_json = "data/tamil_nadu/final/FINAL_HOLDOUT.json"

    # Evaluate Certified Phase 9 Baseline Production Model
    baseline_ckpt_path = "data/tamil_nadu/final/checkpoints/tamil_nadu_phase9_best.pt"
    baseline_model = BaselineTamilNaduTransferNet(num_classes=3).to(device)
    baseline_model.load_state_dict(torch.load(baseline_ckpt_path, map_location=device))

    ds_val = TamilNaduChipDataset(val_json, augment=False)
    val_loader = DataLoader(ds_val, batch_size=32, shuffle=False)

    baseline_val_metrics, baseline_val_probs, baseline_val_logits, val_targets = evaluate_loader(baseline_model, val_loader, device)

    save_confusion_matrix_plot(np.array(baseline_val_metrics["confusion_matrix"]), "Baseline Phase 9 Validation Confusion Matrix", f"{base_dir}/figures/baseline_val_cm.png")

    print("\n--- PHASE 9 BASELINE VALIDATION METRICS ---")
    print(f"Val Accuracy: {baseline_val_metrics['accuracy']}%")
    print(f"Val Macro-F1: {baseline_val_metrics['macro_f1']}")
    print(f"Val Damaged Recall: {baseline_val_metrics['damaged_recall']}% | Destroyed Recall: {baseline_val_metrics['destroyed_recall']}%")
    print(f"Val Calibration ECE: {baseline_val_metrics['calibration_ece']}")

    all_experiment_logs = {}
    all_experiment_logs["Baseline_Phase9"] = baseline_val_metrics

    # Define Experiment Registry (Experiments A - F)
    class_weights_inv = torch.tensor([0.40, 2.64, 10.11]).to(device)

    exp_specs = [
        # Exp A: Loss Functions
        ("ExpA1_WeightedCE", BaselineTamilNaduTransferNet(num_classes=3), False, 0.0, None, nn.CrossEntropyLoss(weight=class_weights_inv), 1e-4, 1e-3, False),
        ("ExpA2_FocalLoss", BaselineTamilNaduTransferNet(num_classes=3), False, 0.0, None, FocalLoss(alpha=torch.tensor([0.2, 1.0, 3.0])), 1e-4, 1e-3, False),
        ("ExpA3_ClassBalancedLoss", BaselineTamilNaduTransferNet(num_classes=3), False, 0.0, None, ClassBalancedLoss(loss_type='cross_entropy'), 1e-4, 1e-3, False),
        ("ExpA4_CB_FocalLoss", BaselineTamilNaduTransferNet(num_classes=3), False, 0.0, None, ClassBalancedLoss(loss_type='focal', gamma=2.0), 1e-4, 1e-3, False),

        # Exp B: Balanced Sampling
        ("ExpB1_WeightedSampler", BaselineTamilNaduTransferNet(num_classes=3), False, 0.0, "weighted_sampler", nn.CrossEntropyLoss(), 1e-4, 1e-3, False),

        # Exp C: SAR Augmentation
        ("ExpC1_SAR_Augmentations", BaselineTamilNaduTransferNet(num_classes=3), True, 0.02, None, nn.CrossEntropyLoss(weight=class_weights_inv), 1e-4, 1e-3, False),
        ("ExpC2_Aug_HighNoise_Shift", BaselineTamilNaduTransferNet(num_classes=3), True, 0.05, None, nn.CrossEntropyLoss(weight=class_weights_inv), 1e-4, 1e-3, False),

        # Exp D: Temporal Change Representation
        ("ExpD1_SignedDiff", SignedDiffTamilNaduNet(num_classes=3), True, 0.02, None, nn.CrossEntropyLoss(weight=class_weights_inv), 1e-4, 1e-3, False),
        ("ExpD2_ConcatTemporal", ConcatTemporalTamilNaduNet(num_classes=3), True, 0.02, None, nn.CrossEntropyLoss(weight=class_weights_inv), 1e-4, 1e-3, False),
        ("ExpD3_LogRatio", LogRatioTamilNaduNet(num_classes=3), True, 0.02, None, nn.CrossEntropyLoss(weight=class_weights_inv), 1e-4, 1e-3, False),
        ("ExpD4_AttentionFusion", AttentionFusionTamilNaduNet(num_classes=3), True, 0.02, None, nn.CrossEntropyLoss(weight=class_weights_inv), 1e-4, 1e-3, False),

        # Exp E: Classifier Head Refinements
        ("ExpE1_LayerNorm_Dropout03", LayerNormTamilNaduNet(num_classes=3, dropout=0.3), True, 0.02, None, nn.CrossEntropyLoss(weight=class_weights_inv), 1e-4, 1e-3, False),
        ("ExpE2_LabelSmoothing01", BaselineTamilNaduTransferNet(num_classes=3), True, 0.02, None, LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights_inv), 1e-4, 1e-3, False),

        # Exp F: Optimization & Stability
        ("ExpF1_CosineScheduler", BaselineTamilNaduTransferNet(num_classes=3), True, 0.02, None, nn.CrossEntropyLoss(weight=class_weights_inv), 1e-4, 1e-3, True),
    ]

    trained_models = {}
    exp_val_records = {}

    for exp_id, model_inst, aug_flag, noise_lvl, sampler_type, loss_fn, lr_enc, lr_head, sched_flag in exp_specs:
        set_seed(42)
        print(f"\n--- Running {exp_id} ---")

        ds_train = TamilNaduChipDataset(train_json, augment=aug_flag, noise_std=noise_lvl)

        if sampler_type == "weighted_sampler":
            class_counts = [4362, 654, 171]
            class_weights = 1.0 / np.array(class_counts)
            sample_weights = [class_weights[lbl] for lbl in ds_train.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(ds_train, batch_size=32, sampler=sampler)
        else:
            train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)

        optimizer = optim.AdamW([
            {'params': model_inst.pre_encoder.parameters(), 'lr': lr_enc},
            {'params': model_inst.post_encoder.parameters(), 'lr': lr_enc},
            {'params': model_inst.fc1.parameters(), 'lr': lr_head},
            {'params': model_inst.fc2.parameters(), 'lr': lr_head},
        ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15) if sched_flag else None

        trained_m, val_m, val_p, val_l, val_t = train_model(
            model_inst, train_loader, val_loader, loss_fn, optimizer, scheduler=scheduler, epochs=15, device=device
        )

        trained_models[exp_id] = trained_m
        exp_val_records[exp_id] = (val_m, val_p, val_l, val_t)
        all_experiment_logs[exp_id] = val_m

        ckpt_save_path = f"{base_dir}/checkpoints/{exp_id}.pt"
        torch.save(trained_m.state_dict(), ckpt_save_path)

        save_confusion_matrix_plot(np.array(val_m["confusion_matrix"]), f"{exp_id} Validation Confusion Matrix", f"{base_dir}/figures/{exp_id}_val_cm.png")

        print(f"  {exp_id} -> Val Macro-F1: {val_m['macro_f1']} | Val Acc: {val_m['accuracy']}% | Damaged Rec: {val_m['damaged_recall']}% | Destroyed Rec: {val_m['destroyed_recall']}%")

    # -------------------------------------------------------------
    # Experiment G: Threshold Analysis (Validation Only)
    # -------------------------------------------------------------
    print("\n--- Running Experiment G: Constrained Threshold Search ---")
    best_thresh, tuned_f1 = tune_validation_thresholds_constrained(baseline_val_probs, val_targets, min_damaged_rec=20.0, min_destroyed_rec=15.0)
    tuned_val_m, _, _, _ = evaluate_loader(baseline_model, val_loader, device, thresholds=best_thresh)
    all_experiment_logs["ExpG_Baseline_Tuned"] = tuned_val_m
    print(f"  ExpG_Baseline_Tuned -> Val Macro-F1: {tuned_val_m['macro_f1']} (Weights: {best_thresh}) | Damaged Rec: {tuned_val_m['damaged_recall']}% | Destroyed Rec: {tuned_val_m['destroyed_recall']}%")

    # -------------------------------------------------------------
    # Experiment H: Temperature Scaling Calibration
    # -------------------------------------------------------------
    print("\n--- Running Experiment H: Temperature Scaling Calibration ---")
    temp_scaler = TemperatureScaler()
    opt_temp = temp_scaler.fit(baseline_val_logits, val_targets)
    calib_val_m, _, _, _ = evaluate_loader(baseline_model, val_loader, device, temperature_scaler=temp_scaler)
    all_experiment_logs["ExpH_Temperature_Calibrated"] = calib_val_m
    print(f"  ExpH_Temperature_Calibrated -> Optimal Temp: {opt_temp:.4f} | ECE: {calib_val_m['calibration_ece']} (Baseline ECE: {baseline_val_metrics['calibration_ece']})")

    # Save Experiment Summary Config & JSON
    with open(f"{base_dir}/results/experiment_validation_summary.json", "w") as f:
        json.dump(all_experiment_logs, f, indent=4)

    # -------------------------------------------------------------
    # Model Selection Rule Evaluation
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("APPLYING STRICT MODEL SELECTION RULE")
    print("=" * 70)

    best_candidate_name = "Baseline_Phase9"
    best_val_macro_f1 = baseline_val_metrics["macro_f1"]
    promotion_decision = False
    selected_model = baseline_model
    selected_thresholds = None
    selected_temp_scaler = None

    print(f"Baseline Phase 9 Val Macro-F1: {baseline_val_metrics['macro_f1']} | Damaged Rec: {baseline_val_metrics['damaged_recall']}% | Destroyed Rec: {baseline_val_metrics['destroyed_recall']}%")

    for name, res in all_experiment_logs.items():
        if name == "Baseline_Phase9":
            continue
        # Candidate selection rule: Must improve Val Macro-F1 AND maintain/improve damage recall
        if res["macro_f1"] > baseline_val_metrics["macro_f1"] + 0.01:
            if res["damaged_recall"] >= baseline_val_metrics["damaged_recall"] or res["destroyed_recall"] >= baseline_val_metrics["destroyed_recall"]:
                if res["macro_f1"] > best_val_macro_f1:
                    best_val_macro_f1 = res["macro_f1"]
                    best_candidate_name = name
                    promotion_decision = True
                    if name in trained_models:
                        selected_model = trained_models[name]
                    elif name == "ExpG_Baseline_Tuned":
                        selected_model = baseline_model
                        selected_thresholds = best_thresh
                    elif name == "ExpH_Temperature_Calibrated":
                        selected_model = baseline_model
                        selected_temp_scaler = temp_scaler

    if promotion_decision:
        print(f"\nPROMOTING CANDIDATE: {best_candidate_name} (Val Macro-F1: {best_val_macro_f1})")

        # Multi-Seed Verification for Candidate (Seeds 42, 43, 44)
        print(f"\nRunning Multi-Seed Stability Verification for {best_candidate_name} (Seeds 42, 43, 44)...")
        seed_scores = []
        for seed in [42, 43, 44]:
            set_seed(seed)
            # Find candidate specification
            for spec in exp_specs:
                if spec[0] == best_candidate_name:
                    m_s = copy.deepcopy(spec[1])
                    ds_tr_s = TamilNaduChipDataset(train_json, augment=spec[2], noise_std=spec[3])
                    tr_ld_s = DataLoader(ds_tr_s, batch_size=32, shuffle=True)
                    opt_s = optim.AdamW(m_s.parameters(), lr=spec[6], weight_decay=1e-4)
                    m_s_tr, m_s_val_m, _, _, _ = train_model(m_s, tr_ld_s, val_loader, spec[5], opt_s, epochs=15, device=device)
                    seed_scores.append(m_s_val_m["macro_f1"])

        mean_seed_f1 = np.mean(seed_scores)
        std_seed_f1 = np.std(seed_scores)
        print(f"  Multi-Seed Val Macro-F1: {mean_seed_f1:.4f} ± {std_seed_f1:.4f}")
    else:
        print("\nDECISION: Retaining certified Phase 9 baseline production model (`tamil_nadu_phase9_best.pt`). Experimental candidates did not beat baseline Macro-F1 while maintaining damage recall.")

    # -------------------------------------------------------------
    # Final Frozen Evaluation on Locked Test and Unseen Holdout Sets
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL FROZEN EVALUATION ON LOCKED TEST & UNSEEN HOLDOUT SETS")
    print("=" * 70)

    ds_test = TamilNaduChipDataset(test_json, augment=False)
    ds_holdout = TamilNaduChipDataset(holdout_json, augment=False)

    test_loader = DataLoader(ds_test, batch_size=32, shuffle=False)
    holdout_loader = DataLoader(ds_holdout, batch_size=32, shuffle=False)

    test_metrics, _, _, _ = evaluate_loader(selected_model, test_loader, device, thresholds=selected_thresholds, temperature_scaler=selected_temp_scaler)
    holdout_metrics, _, _, _ = evaluate_loader(selected_model, holdout_loader, device, thresholds=selected_thresholds, temperature_scaler=selected_temp_scaler)

    save_confusion_matrix_plot(np.array(test_metrics["confusion_matrix"]), "Locked Test Set Confusion Matrix", f"{base_dir}/figures/final_test_cm.png")
    save_confusion_matrix_plot(np.array(holdout_metrics["confusion_matrix"]), "Unseen Holdout Set Confusion Matrix", f"{base_dir}/figures/final_holdout_cm.png")

    print("\n--- LOCKED TEST SET RESULTS (1,112 Samples) ---")
    print(f"Accuracy: {test_metrics['accuracy']}% | 95% CI: {test_metrics['ci_95']}")
    print(f"Macro-F1: {test_metrics['macro_f1']} | Weighted F1: {test_metrics['weighted_f1']} | Balanced Acc: {test_metrics['balanced_accuracy']}%")
    print(f"Recall per class [INTACT, DAMAGED, DESTROYED]: [{test_metrics['intact_recall']}%, {test_metrics['damaged_recall']}%, {test_metrics['destroyed_recall']}%]")
    print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")
    print(f"Calibration ECE: {test_metrics['calibration_ece']}")

    print("\n--- UNSEEN HOLDOUT SET RESULTS (438 Samples) ---")
    print(f"Accuracy: {holdout_metrics['accuracy']}% | 95% CI: {holdout_metrics['ci_95']}")
    print(f"Macro-F1: {holdout_metrics['macro_f1']} | Weighted F1: {holdout_metrics['weighted_f1']} | Balanced Acc: {holdout_metrics['balanced_accuracy']}%")
    print(f"Recall per class [INTACT, DAMAGED, DESTROYED]: [{holdout_metrics['intact_recall']}%, {holdout_metrics['damaged_recall']}%, {holdout_metrics['destroyed_recall']}%]")
    print(f"Confusion Matrix:\n{np.array(holdout_metrics['confusion_matrix'])}")
    print(f"Calibration ECE: {holdout_metrics['calibration_ece']}")

    # Save Final Report JSON
    with open(baseline_ckpt_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    final_results_summary = {
        "production_model_selected": "Baseline_Phase9" if not promotion_decision else best_candidate_name,
        "promotion_occurred": promotion_decision,
        "checkpoint_path": baseline_ckpt_path,
        "checkpoint_sha256": ckpt_sha256,
        "validation_summary": all_experiment_logs,
        "final_locked_test_metrics": test_metrics,
        "final_unseen_holdout_metrics": holdout_metrics
    }

    report_path = f"{base_dir}/results/final_research_results.json"
    with open(report_path, "w") as f:
        json.dump(final_results_summary, f, indent=4)

    print(f"\nSaved final results summary to {report_path}")
    return final_results_summary

if __name__ == "__main__":
    run_full_experiment_suite()
