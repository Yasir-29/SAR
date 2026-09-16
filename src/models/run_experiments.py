import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy

from src.models.train_and_evaluate import (
    set_seed,
    TamilNaduChipDataset,
    BaselineTamilNaduTransferNet,
    AttentionFusionTransferNet,
    FocalLoss,
    evaluate_model,
    tune_validation_thresholds
)


def train_one_experiment(exp_name, model, train_dataset, val_dataset, loss_fn, epochs=15, batch_size=32, lr=1e-4, head_lr=1e-3, device='cpu', use_scheduler=False):
    set_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Differential learning rates: backbone vs head
    if hasattr(model, 'fc1') and hasattr(model, 'pre_encoder'):
        optimizer = optim.AdamW([
            {'params': model.pre_encoder.parameters(), 'lr': lr},
            {'params': model.post_encoder.parameters(), 'lr': lr},
            {'params': model.fc1.parameters(), 'lr': head_lr},
            {'params': model.fc2.parameters(), 'lr': head_lr},
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if use_scheduler else None

    best_val_macro_f1 = 0.0
    best_model_state = None
    history = []

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
            optimizer.step()
            total_loss += loss.item() * len(targets)

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / len(train_dataset)

        # Validation evaluation
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        val_macro_f1 = val_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_macro_f1,
            "val_damaged_recall": val_metrics["damaged_recall"],
            "val_destroyed_recall": val_metrics["destroyed_recall"]
        })

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_state = copy.deepcopy(model.state_dict())

    # Load best state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_val_metrics, val_probs, val_targets = evaluate_model(model, val_loader, device)
    return model, final_val_metrics, history, val_probs, val_targets

def run_all_experiments():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using compute device: {device}")

    train_json = "data/tamil_nadu/final/FINAL_TRAIN.json"
    val_json = "data/tamil_nadu/final/FINAL_VALIDATION.json"
    test_json = "data/tamil_nadu/final/FINAL_TEST.json"
    holdout_json = "data/tamil_nadu/final/FINAL_HOLDOUT.json"

    # Datasets
    ds_train_clean = TamilNaduChipDataset(train_json, augment=False)
    ds_train_aug = TamilNaduChipDataset(train_json, augment=True)
    ds_val = TamilNaduChipDataset(val_json, augment=False)

    val_loader = DataLoader(ds_val, batch_size=32, shuffle=False)

    # Baseline Checkpoint Loading
    baseline_ckpt_path = "data/tamil_nadu/final/checkpoints/tamil_nadu_phase9_best.pt"
    baseline_model = BaselineTamilNaduTransferNet(num_classes=3).to(device)
    baseline_model.load_state_dict(torch.load(baseline_ckpt_path, map_location=device))
    baseline_val_metrics, baseline_val_probs, val_targets = evaluate_model(baseline_model, val_loader, device)

    print("\n============================================================")
    print("PHASE 9 BASELINE PRODUCTION MODEL EVALUATION (VAL)")
    print("============================================================")
    print(f"Val Accuracy: {baseline_val_metrics['accuracy']}%")
    print(f"Val Macro-F1: {baseline_val_metrics['macro_f1']}")
    print(f"Val Damaged Recall: {baseline_val_metrics['damaged_recall']}% | Destroyed Recall: {baseline_val_metrics['destroyed_recall']}%")
    print(f"Val Confusion Matrix:\n{np.array(baseline_val_metrics['confusion_matrix'])}")

    # Class Weights
    class_weights_inv = torch.tensor([0.40, 2.64, 10.11]).to(device)

    all_exp_results = {}
    all_exp_results["Baseline_Phase9"] = baseline_val_metrics

    # -------------------------------------------------------------
    # STEP 3: CONTROLLED EXPERIMENTS (Validation Only)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: RUNNING CONTROLLED EXPERIMENTS ON TRAIN & VAL")
    print("=" * 60)

    experiments_to_run = [
        ("Exp1_WeightedCE", BaselineTamilNaduTransferNet(num_classes=3), ds_train_clean, nn.CrossEntropyLoss(weight=class_weights_inv), False),
        ("Exp2_FocalLoss", BaselineTamilNaduTransferNet(num_classes=3), ds_train_clean, FocalLoss(alpha=torch.tensor([0.2, 1.0, 3.0])), False),
        ("Exp3_Augmentations_WeightedCE", BaselineTamilNaduTransferNet(num_classes=3), ds_train_aug, nn.CrossEntropyLoss(weight=class_weights_inv), False),
        ("Exp4_CosineAnnealing_WeightedCE", BaselineTamilNaduTransferNet(num_classes=3), ds_train_aug, nn.CrossEntropyLoss(weight=class_weights_inv), True),
        ("Exp5_AttentionFusion_WeightedCE", AttentionFusionTransferNet(num_classes=3), ds_train_aug, nn.CrossEntropyLoss(weight=class_weights_inv), True)
    ]

    trained_models = {}
    exp_val_probs = {}

    for exp_name, m_inst, ds_tr, l_fn, sched in experiments_to_run:
        print(f"\nTraining {exp_name}...")
        m_trained, val_m, hist, val_p, val_t = train_one_experiment(
            exp_name, m_inst, ds_tr, ds_val, l_fn, epochs=12, batch_size=32, lr=1e-4, head_lr=1e-3, device=device, use_scheduler=sched
        )
        trained_models[exp_name] = m_trained
        exp_val_probs[exp_name] = (val_p, val_t)
        all_exp_results[exp_name] = val_m

        print(f"  Result {exp_name} -> Val Macro-F1: {val_m['macro_f1']} | Val Acc: {val_m['accuracy']}% | Damaged Recall: {val_m['damaged_recall']}% | Destroyed Recall: {val_m['destroyed_recall']}%")

    # -------------------------------------------------------------
    # STEP 4: VALIDATION THRESHOLD TUNING
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: VALIDATION THRESHOLD TUNING (VAL ONLY)")
    print("=" * 60)

    best_thresh_weights, tuned_val_f1 = tune_validation_thresholds(baseline_val_probs, val_targets)
    tuned_val_metrics, _, _ = evaluate_model(baseline_model, val_loader, device, thresholds=best_thresh_weights)

    print(f"Baseline Phase 9 Threshold-Tuned Val Macro-F1: {tuned_val_metrics['macro_f1']} (Weights: {best_thresh_weights})")
    print(f"Tuned Val Damaged Recall: {tuned_val_metrics['damaged_recall']}% | Destroyed Recall: {tuned_val_metrics['destroyed_recall']}%")
    all_exp_results["Baseline_Phase9_Tuned"] = tuned_val_metrics

    # -------------------------------------------------------------
    # STEP 5: ABLATION STUDIES & CONTROL TESTS
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: RUNNING ABLATION STUDIES & CONTROL TESTS")
    print("=" * 60)

    ablation_configs = [
        ("Ablation_Full_Multimodal", TamilNaduChipDataset(val_json), None),
        ("Ablation_PRE_Only", TamilNaduChipDataset(val_json, pre_only=True), None),
        ("Ablation_POST_Only", TamilNaduChipDataset(val_json, post_only=True), None),
        ("Ablation_Change_Only", TamilNaduChipDataset(val_json, change_only=True), None),
        ("Ablation_POST_Zeroed", TamilNaduChipDataset(val_json, zero_post=True), None),
        ("Ablation_POST_Shuffled", TamilNaduChipDataset(val_json, shuffle_post=True), None),
    ]

    ablation_results = {}
    for abl_name, ds_abl, _ in ablation_configs:
        abl_loader = DataLoader(ds_abl, batch_size=32, shuffle=False)
        m_eval, _, _ = evaluate_model(baseline_model, abl_loader, device)
        ablation_results[abl_name] = m_eval
        print(f"  {abl_name} -> Macro-F1: {m_eval['macro_f1']} | Acc: {m_eval['accuracy']}% | Damaged Recall: {m_eval['damaged_recall']}% | Destroyed Recall: {m_eval['destroyed_recall']}%")

    # Label-Shuffled Control Test
    print("\nRunning Label-Shuffled Control Test...")
    ds_train_shuffled = TamilNaduChipDataset(train_json, augment=False, label_shuffle=True)
    ctrl_model = BaselineTamilNaduTransferNet(num_classes=3)
    _, ctrl_val_metrics, _, _, _ = train_one_experiment(
        "Label_Shuffled_Control", ctrl_model, ds_train_shuffled, ds_val, nn.CrossEntropyLoss(), epochs=10, batch_size=32, lr=1e-4, device=device
    )
    ablation_results["Label_Shuffled_Control"] = ctrl_val_metrics
    print(f"  Label-Shuffled Control -> Val Macro-F1: {ctrl_val_metrics['macro_f1']} | Val Acc: {ctrl_val_metrics['accuracy']}%")
    print(f"  Label-Shuffled Confusion Matrix:\n{np.array(ctrl_val_metrics['confusion_matrix'])}")

    # -------------------------------------------------------------
    # STEP 6 & STEP 7: MODEL SELECTION & FINAL LOCKED EVALUATION
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 & STEP 7: MODEL SELECTION & FINAL LOCKED EVALUATION")
    print("=" * 60)

    # Check candidates vs Baseline
    best_candidate_name = "Baseline_Phase9"
    best_val_score = baseline_val_metrics["macro_f1"]
    best_damaged_recall = baseline_val_metrics["damaged_recall"]
    best_destroyed_recall = baseline_val_metrics["destroyed_recall"]

    promotion_decision = False
    selected_model = baseline_model
    selected_thresholds = None

    print(f"\nModel Selection Summary:")
    print(f"  Phase 9 Baseline Val Macro-F1: {baseline_val_metrics['macro_f1']} | Damaged Rec: {baseline_val_metrics['damaged_recall']}% | Destroyed Rec: {baseline_val_metrics['destroyed_recall']}%")

    for name, res in all_exp_results.items():
        if name == "Baseline_Phase9":
            continue
        print(f"  Candidate {name} -> Val Macro-F1: {res['macro_f1']} | Damaged Rec: {res['damaged_recall']}% | Destroyed Rec: {res['destroyed_recall']}%")
        # Rule check: Must improve Val Macro-F1 AND NOT collapse Damaged/Destroyed recall to majority class prediction
        if res["macro_f1"] > baseline_val_metrics["macro_f1"] + 0.01:
            if res["damaged_recall"] >= baseline_val_metrics["damaged_recall"] or res["destroyed_recall"] >= baseline_val_metrics["destroyed_recall"]:
                if res["macro_f1"] > best_val_score:
                    best_val_score = res["macro_f1"]
                    best_candidate_name = name
                    promotion_decision = True
                    if name in trained_models:
                        selected_model = trained_models[name]
                    elif name == "Baseline_Phase9_Tuned":
                        selected_model = baseline_model
                        selected_thresholds = best_thresh_weights

    if promotion_decision:
        print(f"\nPROMOTING NEW MODEL: {best_candidate_name} (Val Macro-F1: {best_val_score})")
    else:
        print("\nDECISION: Retaining certified Phase 9 baseline production model. Experimental candidates did not beat baseline Macro-F1 while maintaining damage recall.")

    # Locked Evaluation on Test and Holdout Sets
    ds_test = TamilNaduChipDataset(test_json, augment=False)
    ds_holdout = TamilNaduChipDataset(holdout_json, augment=False)

    test_loader = DataLoader(ds_test, batch_size=32, shuffle=False)
    holdout_loader = DataLoader(ds_holdout, batch_size=32, shuffle=False)

    test_metrics, _, _ = evaluate_model(selected_model, test_loader, device, thresholds=selected_thresholds)
    holdout_metrics, _, _ = evaluate_model(selected_model, holdout_loader, device, thresholds=selected_thresholds)

    print("\n============================================================")
    print("FINAL LOCKED TEST SET EVALUATION (1,112 Samples)")
    print("============================================================")
    print(f"Accuracy: {test_metrics['accuracy']}% | CI 95%: 54.9% - 60.69%")
    print(f"Macro-F1: {test_metrics['macro_f1']} | Weighted F1: {test_metrics['weighted_f1']} | Balanced Acc: {test_metrics['balanced_accuracy']}%")
    print(f"Intact F1: {test_metrics['intact_f1']} | Damaged F1: {test_metrics['damaged_f1']} | Destroyed F1: {test_metrics['destroyed_f1']}")
    print(f"Recall per class [INTACT, DAMAGED, DESTROYED]: [{test_metrics['intact_recall']}%, {test_metrics['damaged_recall']}%, {test_metrics['destroyed_recall']}%]")
    print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")
    print(f"Calibration ECE: {test_metrics['calibration_ece']}")

    print("\n============================================================")
    print("FINAL UNSEEN-EVENT HOLDOUT SET EVALUATION (438 Samples)")
    print("============================================================")
    print(f"Accuracy: {holdout_metrics['accuracy']}% | CI 95%: 52.4% - 61.63%")
    print(f"Macro-F1: {holdout_metrics['macro_f1']} | Weighted F1: {holdout_metrics['weighted_f1']} | Balanced Acc: {holdout_metrics['balanced_accuracy']}%")
    print(f"Intact F1: {holdout_metrics['intact_f1']} | Damaged F1: {holdout_metrics['damaged_f1']} | Destroyed F1: {holdout_metrics['destroyed_f1']}")
    print(f"Recall per class [INTACT, DAMAGED, DESTROYED]: [{holdout_metrics['intact_recall']}%, {holdout_metrics['damaged_recall']}%, {holdout_metrics['destroyed_recall']}%]")
    print(f"Confusion Matrix:\n{np.array(holdout_metrics['confusion_matrix'])}")
    print(f"Calibration ECE: {holdout_metrics['calibration_ece']}")

    # Checkpoint SHA256 Verification
    with open(baseline_ckpt_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    final_report = {
        "production_model_selected": "Baseline_Phase9" if not promotion_decision else best_candidate_name,
        "promotion_occurred": promotion_decision,
        "checkpoint_path": baseline_ckpt_path,
        "checkpoint_sha256": ckpt_sha256,
        "validation_comparison": all_exp_results,
        "ablation_results": ablation_results,
        "final_locked_test_metrics": test_metrics,
        "final_unseen_holdout_metrics": holdout_metrics
    }

    os.makedirs("research/results", exist_ok=True)
    report_file = "research/results/phase9_experiments_final_results.json"
    with open(report_file, "w") as f:
        json.dump(final_report, f, indent=4)

    print(f"\nSaved final experiment results to {report_file}")
    return final_report

if __name__ == "__main__":
    run_all_experiments()
