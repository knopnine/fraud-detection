import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from fraud_features import FraudFeatures
from improve_fraud import candidate_models, grouped_splits, select_threshold


class ImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_limit = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.thread_limit.restore_original_limits()

    def test_merchant_mean_uses_fit_rows_only_and_unseen_fallback(self):
        train = pd.DataFrame({"amount": [10., 30., 80., 90.], "merchant_type": ["a", "a", "b", "b"],
                              "device_type": ["mobile"] * 4})
        held_out = pd.DataFrame({"amount": [1000000., 20.], "merchant_type": ["a", "unknown"],
                                "device_type": ["desktop", "desktop"]})
        transformer = FraudFeatures().fit(train)
        known_mean = np.mean(np.log1p([10., 30.]))
        output = transformer.transform(held_out)
        self.assertAlmostEqual(transformer.merchant_means_["a"], known_mean)
        self.assertAlmostEqual(output.merchant_log_deviation.iloc[0], np.log1p(1000000.) - known_mean)
        self.assertAlmostEqual(output.merchant_log_deviation.iloc[1], np.log1p(20.) - np.log1p(train.amount).mean())
        self.assertEqual(output.pair_frequency.iloc[0], 0)
        self.assertTrue(np.isfinite(output.select_dtypes(include="number")).all().all())
        self.assertEqual(list(train.columns), ["amount", "merchant_type", "device_type"])

    def test_inner_and_outer_group_separation(self):
        y = np.tile([0, 0, 0, 0, 1], 40)
        groups = np.arange(200) // 2
        for fit, outer in grouped_splits(y, groups, 5, 42):
            self.assertFalse(set(groups[fit]) & set(groups[outer]))
            for train, validate in grouped_splits(y[fit], groups[fit], 3, 43):
                self.assertFalse(set(groups[fit][train]) & set(groups[fit][validate]))
                self.assertFalse(set(groups[fit][train]) & set(groups[outer]))

    def test_budget_and_tied_score_threshold(self):
        y = np.array([0] * 95 + [1] * 5)
        scores = np.linspace(0, 1, 100)
        threshold = select_threshold(y, scores, .1)
        self.assertLessEqual((scores >= threshold).mean(), .1)
        scores[:] = .05
        threshold = select_threshold(y, scores, .1)
        self.assertEqual((scores >= threshold).sum(), 0)

    def test_all_candidates_fit_and_handle_unseen_categories(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame({"amount": rng.uniform(1, 200, 160),
                          "merchant_type": np.tile(["travel", "clothing"], 80),
                          "device_type": np.tile(["mobile", "desktop", "tablet", "mobile"], 40)})
        y = np.zeros(160, dtype=int)
        y[rng.choice(160, 16, replace=False)] = 1
        new = pd.DataFrame({"amount": [25., 125.], "merchant_type": ["new", "travel"],
                            "device_type": ["new", "mobile"]})
        for name, estimator in candidate_models().items():
            with self.subTest(model=name):
                fitted = clone(estimator).fit(X, y)
                scores = fitted.predict_proba(new)[:, 1]
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((scores >= 0) & (scores <= 1)).all())


if __name__ == "__main__":
    unittest.main()
