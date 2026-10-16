"""Training-only feature statistics for the fraud learning experiment."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FraudFeatures(TransformerMixin, BaseEstimator):
    """Add row-local features and reference statistics learned only during fit."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.global_log_mean_ = float(np.log1p(X.amount).mean())
        logs = np.log1p(X.amount)
        self.merchant_means_ = logs.groupby(X.merchant_type).mean().to_dict()
        self.device_means_ = logs.groupby(X.device_type).mean().to_dict()
        pairs = list(zip(X.merchant_type, X.device_type))
        self.pair_frequencies_ = pd.Series(pairs).value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        check_is_fitted(self, "merchant_means_")
        result = X.copy()
        logs = np.log1p(X.amount)
        result["amount_category"] = pd.cut(
            X.amount, [-np.inf, 25, 100, 500, np.inf],
            labels=["low", "medium", "high", "very_high"], right=False,
        ).astype(str)
        result["is_round_amount"] = np.isclose(X.amount, np.round(X.amount), atol=1e-8, rtol=0).astype(int)
        result["merchant_device_combo"] = X.merchant_type + "_" + X.device_type
        result["merchant_log_deviation"] = logs - X.merchant_type.map(
            self.merchant_means_).fillna(self.global_log_mean_)
        result["device_log_deviation"] = logs - X.device_type.map(
            self.device_means_).fillna(self.global_log_mean_)
        result["pair_frequency"] = [
            self.pair_frequencies_.get(pair, 0.0)
            for pair in zip(X.merchant_type, X.device_type)
        ]
        return result
