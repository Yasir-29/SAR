import os
import json
import hashlib
import numpy as np
import torch

def audit_dataset_quality():
    print("=" * 60)
    print("STEP 2: RUNNING COMPLETE DATASET QUALITY & LEAKAGE AUDIT")
    print("=" * 60)

    split_files = {
        "train": "data/tamil_nadu/final/FINAL_TRAIN.json",
        "val": "data/tamil_nadu/final/FINAL_VALIDATION.json",
        "test": "data/tamil_nadu/final/FINAL_TEST.json",
        "holdout": "data/tamil_nadu/final/FINAL_HOLDOUT.json"
    }

    splits_data = {}
    split_building_ids = {}
    split_event_ids = {}
    class_distributions = {}
    corrupt_chips = []
    blank_chips = []
    chip_hashes = {}
    duplicate_chips = []

    for split_name, path in split_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        splits_data[split_name] = data
        building_ids = set()
        event_ids = set()
        class_counts = {0: 0, 1: 0, 2: 0}

        for idx, item in enumerate(data):
            b_id = item["building_id"]
            e_id = item.get("event_id", item.get("event_name", "unknown"))
            lbl = item["verified_label"]
            building_ids.add(b_id)
            event_ids.add(e_id)
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

            # Check chip existence & corruption
            sar_pre_path = item["sar_pre_path"]
            sar_post_path = item["sar_post_path"]

            for chip_type, chip_path in [("pre", sar_pre_path), ("post", sar_post_path)]:
                if not os.path.exists(chip_path):
                    corrupt_chips.append({"split": split_name, "building_id": b_id, "path": chip_path, "reason": "File missing"})
                    continue
                try:
                    arr = np.load(chip_path)
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        corrupt_chips.append({"split": split_name, "building_id": b_id, "path": chip_path, "reason": "NaN or Inf values"})
                    if arr.std() == 0:
                        blank_chips.append({"split": split_name, "building_id": b_id, "path": chip_path})
                    # Hashing chip array for duplicate detection
                    ch_hash = hashlib.md5(arr.tobytes()).hexdigest()
                    if ch_hash in chip_hashes:
                        duplicate_chips.append({"split": split_name, "building_id": b_id, "path": chip_path, "duplicate_of": chip_hashes[ch_hash]})
                    else:
                        chip_hashes[ch_hash] = (split_name, b_id, chip_path)
                except Exception as e:
                    corrupt_chips.append({"split": split_name, "building_id": b_id, "path": chip_path, "reason": str(e)})

        split_building_ids[split_name] = building_ids
        split_event_ids[split_name] = event_ids
        class_distributions[split_name] = class_counts

    # 1. Unique building IDs check
    all_b_ids = []
    for b_set in split_building_ids.values():
        all_b_ids.extend(list(b_set))
    unique_b_ids = set(all_b_ids)
    total_samples = len(all_b_ids)
    print(f"Total building samples: {total_samples}")
    print(f"Unique building IDs: {len(unique_b_ids)}")

    # 2. Overlap checks
    train_val_overlap = len(split_building_ids["train"].intersection(split_building_ids["val"]))
    train_test_overlap = len(split_building_ids["train"].intersection(split_building_ids["test"]))
    train_holdout_overlap = len(split_building_ids["train"].intersection(split_building_ids["holdout"]))
    val_test_overlap = len(split_building_ids["val"].intersection(split_building_ids["test"]))
    val_holdout_overlap = len(split_building_ids["val"].intersection(split_building_ids["holdout"]))
    test_holdout_overlap = len(split_building_ids["test"].intersection(split_building_ids["holdout"]))

    overlap_summary = {
        "train_val": train_val_overlap,
        "train_test": train_test_overlap,
        "train_holdout": train_holdout_overlap,
        "val_test": val_test_overlap,
        "val_holdout": val_holdout_overlap,
        "test_holdout": test_holdout_overlap
    }

    # 3. Event isolation check for holdout
    train_val_test_events = split_event_ids["train"].union(split_event_ids["val"]).union(split_event_ids["test"])
    holdout_events = split_event_ids["holdout"]
    event_overlap = train_val_test_events.intersection(holdout_events)

    print("\n--- Split Sizes ---")
    for s_name, s_ids in split_building_ids.items():
        print(f"  {s_name}: {len(s_ids)} samples | Classes: {class_distributions[s_name]}")

    print("\n--- Overlap Audit ---")
    for k, v in overlap_summary.items():
        print(f"  {k} overlap: {v}")

    print(f"\nUnseen Holdout Events: {list(holdout_events)}")
    print(f"Holdout Event Overlap with Train/Val/Test: {list(event_overlap)} (Count: {len(event_overlap)})")

    print(f"\nCorrupt Chips: {len(corrupt_chips)}")
    print(f"Blank Chips: {len(blank_chips)}")

    audit_result = {
        "total_samples": total_samples,
        "unique_building_ids": len(unique_b_ids),
        "split_counts": {k: len(v) for k, v in split_building_ids.items()},
        "class_distributions": class_distributions,
        "overlap_audit": overlap_summary,
        "holdout_events": list(holdout_events),
        "holdout_event_overlap_count": len(event_overlap),
        "corrupt_chip_count": len(corrupt_chips),
        "blank_chip_count": len(blank_chips),
        "duplicate_chip_count": len(duplicate_chips),
        "data_quality_status": "PASS" if sum(overlap_summary.values()) == 0 and len(corrupt_chips) == 0 else "FAIL"
    }

    os.makedirs("research/audit_reports", exist_ok=True)
    out_path = "research/audit_reports/dataset_quality_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit_result, f, indent=4)
    print(f"\nSaved Dataset Quality Audit to {out_path}")
    return audit_result

if __name__ == "__main__":
    audit_dataset_quality()
