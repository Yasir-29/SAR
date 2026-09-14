#!/usr/bin/env python3
"""
TEST SUITE FOR CMD DISASTER EVENT & COORDINATE PREDICTION TOOL
Verifies end-to-end functionality of scripts/predict_event_coordinate.py
"""

import os
import sys
import hashlib
import unittest
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.predict_event_coordinate import (
    DisasterPredictionCLI,
    verify_checkpoint_sha256,
    validate_coordinates,
    CHECKPOINT_PATH,
    EXPECTED_SHA256,
    CLASS_NAMES
)


class TestCLIPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n============================================================")
        print("RUNNING CLI PREDICTION SUITE (PHASE 9 MODEL VERIFICATION)")
        print("============================================================")
        cls.cli = DisasterPredictionCLI()

    def test_1_checkpoint_exists(self):
        """Test 1: Checkpoint file exists."""
        self.assertTrue(os.path.exists(CHECKPOINT_PATH), f"Checkpoint missing at {CHECKPOINT_PATH}")
        print("[PASS] Test 1: Checkpoint file exists.")

    def test_2_sha256_matches(self):
        """Test 2: Checkpoint SHA256 matches certified hash exactly."""
        res = verify_checkpoint_sha256(CHECKPOINT_PATH, EXPECTED_SHA256)
        self.assertTrue(res, "Checkpoint SHA256 checksum mismatch!")
        print("[PASS] Test 2: Checkpoint SHA256 matches.")

    def test_3_model_loads(self):
        """Test 3: Phase 9 model loads successfully."""
        self.assertIsNotNone(self.cli.model, "Model instance is None!")
        self.assertIsNotNone(self.cli.device, "Device is None!")
        print(f"[PASS] Test 3: Model loaded on {self.cli.device_name}.")

    def test_4_model_eval_mode(self):
        """Test 4: Model is set to evaluation mode (model.eval())."""
        self.assertFalse(self.cli.model.training, "Model is in training mode! Must be in eval mode.")
        print("[PASS] Test 4: Model is in evaluation mode (model.eval()).")

    def test_5_event_list_valid(self):
        """Test 5: Disaster event list is valid and non-empty."""
        events = list(self.cli.events_map.keys())
        self.assertGreater(len(events), 0, "No disaster events found in registry!")
        self.assertIn("TN_SOUTHERN_FLOOD_2023", events, "Expected event TN_SOUTHERN_FLOOD_2023 not in event list.")
        print(f"[PASS] Test 5: Event list valid ({len(events)} real events found).")

    def test_6_valid_coordinate_accepted(self):
        """Test 6: Valid coordinates are accepted."""
        valid, lat, lon, err = validate_coordinates("13.0429", "80.2486")
        self.assertTrue(valid, f"Valid coordinate rejected: {err}")
        self.assertEqual(lat, 13.0429)
        self.assertEqual(lon, 80.2486)
        print("[PASS] Test 6: Valid coordinate accepted.")

    def test_7_invalid_coordinate_rejected(self):
        """Test 7: Invalid coordinates are rejected cleanly."""
        # Out of range lat
        valid1, _, _, err1 = validate_coordinates("95.0", "80.0")
        self.assertFalse(valid1, "Lat > 90 was accepted!")

        # Out of range lon
        valid2, _, _, err2 = validate_coordinates("13.0", "-190.0")
        self.assertFalse(valid2, "Lon < -180 was accepted!")

        # Non-numeric
        valid3, _, _, err3 = validate_coordinates("abc", "80.0")
        self.assertFalse(valid3, "Non-numeric lat was accepted!")

        print("[PASS] Test 7: Invalid coordinates rejected cleanly.")

    def test_8_building_lookup_works(self):
        """Test 8: Spatial building lookup finds real building for real event."""
        # Find first building item for TN_SOUTHERN_FLOOD_2023
        ev_id = "TN_SOUTHERN_FLOOD_2023"
        ev_info = self.cli.events_map[ev_id]
        first_idx = ev_info["building_indices"][0]
        real_bld = self.cli.service.spatial_building_list[first_idx]

        found, matched_bld, dist_m, _ = self.cli.find_nearest_building_for_event(
            ev_id, real_bld["lat"], real_bld["lon"], search_radius_m=100.0
        )
        self.assertTrue(found, "Building lookup failed for exact real coordinate!")
        self.assertEqual(matched_bld["building_id"], real_bld["building_id"])
        self.assertLessEqual(dist_m, 1.0)
        print(f"[PASS] Test 8: Building lookup works (Found {matched_bld['building_id']} at {dist_m}m).")

    def test_9_prediction_class_valid(self):
        """Test 9: Prediction returns one of INTACT, DAMAGED, DESTROYED."""
        ev_id = "TN_SOUTHERN_FLOOD_2023"
        ev_info = self.cli.events_map[ev_id]
        first_idx = ev_info["building_indices"][0]
        real_bld = self.cli.service.spatial_building_list[first_idx]

        pred_res = self.cli.run_inference(real_bld)
        self.assertIn(pred_res["predicted_class"], CLASS_NAMES, f"Invalid class {pred_res['predicted_class']}")
        print(f"[PASS] Test 9: Prediction class valid ({pred_res['predicted_class']}).")

    def test_10_probabilities_sum_to_one(self):
        """Test 10: Class probabilities sum approximately to 1.0."""
        ev_id = "TN_SOUTHERN_FLOOD_2023"
        ev_info = self.cli.events_map[ev_id]
        first_idx = ev_info["building_indices"][0]
        real_bld = self.cli.service.spatial_building_list[first_idx]

        pred_res = self.cli.run_inference(real_bld)
        probs_sum = sum(pred_res["probabilities"].values())
        self.assertAlmostEqual(probs_sum, 1.0, places=4, msg="Probabilities do not sum to 1.0")
        print(f"[PASS] Test 10: Probabilities sum to 1.0 (Sum = {probs_sum:.6f}).")

    def test_11_confidence_range_valid(self):
        """Test 11: Confidence is between 0.0 and 1.0."""
        ev_id = "TN_SOUTHERN_FLOOD_2023"
        ev_info = self.cli.events_map[ev_id]
        first_idx = ev_info["building_indices"][0]
        real_bld = self.cli.service.spatial_building_list[first_idx]

        pred_res = self.cli.run_inference(real_bld)
        conf = pred_res["confidence"]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        print(f"[PASS] Test 11: Confidence valid ({conf * 100.0:.2f}%).")


if __name__ == "__main__":
    unittest.main()
