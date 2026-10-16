import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fraud_pipeline import choose_threshold, feature_groups, make_splits, metrics, read_data


class PipelineTests(unittest.TestCase):
    def test_duplicate_profiles_cannot_cross_folds(self):
        original = pd.DataFrame({"amount": np.arange(100),
                                 "merchant_type": ["travel"] * 100,
                                 "device_type": ["mobile"] * 100})
        frame = pd.concat([original, original], ignore_index=True)
        y = np.tile(np.arange(100) % 5 == 0, 2).astype(int)
        groups = feature_groups(frame)
        validation_rows = []
        for fit, validate in make_splits(y, groups):
            self.assertFalse(set(groups[fit]) & set(groups[validate]))
            validation_rows.extend(validate)
        self.assertEqual(sorted(validation_rows), list(range(200)))

    def test_threshold_matches_exhaustive_f2_search(self):
        y = np.array([0, 1, 0, 1, 0, 0])
        scores = np.array([0.1, 0.3, 0.4, 0.7, 0.8, 0.9])
        chosen = choose_threshold(y, scores)
        best = max(metrics(y, scores, t)["f2"] for t in np.unique(scores))
        self.assertAlmostEqual(metrics(y, scores, chosen)["f2"], best)

    def test_accuracy_baseline_does_not_hide_missed_fraud(self):
        y = np.array([0] * 95 + [1] * 5)
        result = metrics(y, np.full(100, 0.05), 0.5)
        self.assertEqual(result["accuracy"], 0.95)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["false_negatives"], 5)

    def test_invalid_data_rejected(self):
        base = {"transaction_id": [1, 2], "amount": [10, 20],
                "merchant_type": ["travel", "others"],
                "device_type": ["mobile", "tablet"], "label": [0, 1]}
        for column, bad in [("amount", [-1, 20]), ("amount", [np.inf, 20]),
                            ("label", [0, 2]), ("transaction_id", [1, 1]),
                            ("merchant_type", [" ", "others"])]:
            frame = pd.DataFrame({**base, column: bad})
            with self.subTest(column=column, bad=bad):
                with patch("fraud_pipeline.pd.read_csv", return_value=frame):
                    with self.assertRaises(ValueError):
                        read_data(Path("invalid.csv"))


if __name__ == "__main__":
    unittest.main()
