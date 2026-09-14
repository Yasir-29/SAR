import os
import sys
import hashlib
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.run_final_transfer_learning_audit import TamilNaduTransferNet

EXPECTED_SHA256 = "82eaaf8ddd021652211e544799346abb3a7c94b20e29420a4c1866186114fbbd"
CHECKPOINT_PATH = "data/tamil_nadu/final/checkpoints/tamil_nadu_phase9_best.pt"
CLASS_MAPPING = {0: "INTACT", 1: "DAMAGED", 2: "DESTROYED"}

def test_production_model():
    print("=" * 60)
    print("RUNNING PRODUCTION MODEL VERIFICATION SUITE")
    print("=" * 60)

    # 1. Checkpoint file existence
    assert os.path.exists(CHECKPOINT_PATH), f"FAIL: Checkpoint not found at {CHECKPOINT_PATH}"
    print(f"[PASS] 1. Checkpoint exists: {CHECKPOINT_PATH}")

    # 2. Checkpoint SHA256 verification
    with open(CHECKPOINT_PATH, "rb") as f:
        file_bytes = f.read()
        calc_sha256 = hashlib.sha256(file_bytes).hexdigest()

    assert calc_sha256 == EXPECTED_SHA256, f"FAIL: SHA256 mismatch! Got {calc_sha256}, expected {EXPECTED_SHA256}"
    print(f"[PASS] 2. Checkpoint SHA256 matches: {calc_sha256}")

    # 3. Model load
    device = torch.device("cpu")
    model = TamilNaduTransferNet(num_classes=3).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("[PASS] 3. Model loaded successfully into evaluation mode.")

    # 4. Valid inference pass
    dummy_s1_pre = torch.randn(1, 2, 32, 32)
    dummy_s1_post = torch.randn(1, 2, 32, 32)

    with torch.no_grad():
        logits = model(dummy_s1_pre, dummy_s1_post)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    print("[PASS] 4. Executed inference pass successfully.")

    # 5. Output shape & classes count check
    assert len(probs) == 3, f"FAIL: Expected 3 output classes, got {len(probs)}"
    print(f"[PASS] 5. Output has exactly {len(probs)} classes.")

    # 6. Probabilities sum check (~1.0)
    prob_sum = float(np.sum(probs))
    assert abs(prob_sum - 1.0) < 1e-4, f"FAIL: Probabilities do not sum to 1.0 (got {prob_sum})"
    print(f"[PASS] 6. Probabilities sum to approximately 1.0 (got {prob_sum:.6f})")

    # 7. Predicted class label validity
    pred_class_idx = int(np.argmax(probs))
    pred_label = CLASS_MAPPING.get(pred_class_idx)
    assert pred_label in ["INTACT", "DAMAGED", "DESTROYED"], f"FAIL: Invalid predicted class: {pred_label}"
    print(f"[PASS] 7. Predicted class index {pred_class_idx} maps to valid label: '{pred_label}'")

    # 8. Confidence range check [0, 1]
    confidence = float(np.max(probs))
    assert 0.0 <= confidence <= 1.0, f"FAIL: Confidence {confidence} outside [0, 1]"
    print(f"[PASS] 8. Prediction confidence is valid: {confidence:.4f}")

    print("\n" + "=" * 60)
    print("ALL PRODUCTION MODEL TESTS PASSED CLEANLY [PASS]")
    print("=" * 60)

if __name__ == "__main__":
    test_production_model()
