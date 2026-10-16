"""Train, evaluate, and use a classifier from the supplied transaction CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, fbeta_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FEATURES = ["amount", "merchant_type", "device_type"]
CATEGORIES = ["merchant_type", "device_type"]
SEED = 42


def read_data(path: Path, training: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = FEATURES + (["transaction_id", "label"] if training else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Required columns must contain rows and no missing values.")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="raise")
    if not np.isfinite(frame["amount"]).all() or (frame["amount"] < 0).any():
        raise ValueError("amount must contain finite, nonnegative numbers.")
    for column in CATEGORIES:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{column} contains blank categories.")
    if training:
        if set(frame["label"].unique()) != {0, 1}:
            raise ValueError("Training labels must contain exactly 0 and 1; 1 is treated as fraud.")
        frame["label"] = frame["label"].astype(int)
        if frame["transaction_id"].duplicated().any():
            raise ValueError("transaction_id must be unique in training data.")
    return frame


def feature_groups(frame: pd.DataFrame) -> np.ndarray:
    # Group equal predictors even when IDs or labels differ.
    return frame.groupby(FEATURES, sort=True, dropna=False).ngroup().to_numpy()


def make_splits(y, groups, seed=SEED):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for fit, validate in splits:
        if set(groups[fit]) & set(groups[validate]):
            raise AssertionError("Feature groups overlap across a split.")
        if len(np.unique(np.asarray(y)[fit])) != 2 or len(np.unique(np.asarray(y)[validate])) != 2:
            raise ValueError("Too few independent fraud groups for five-fold evaluation.")
    return splits


def candidates():
    amount = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    prep = ColumnTransformer([
        ("amount", amount, ["amount"]),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORIES),
    ])
    models = {"dummy_prior": DummyClassifier(strategy="prior")}
    for c in (0.1, 1.0):
        for weight in (None, "balanced"):
            models[f"logistic_C{c}_{weight or 'unweighted'}"] = LogisticRegression(
                C=c, class_weight=weight, max_iter=2000, random_state=SEED,
            )
    for weight in (None, "balanced"):
        models[f"forest_{weight or 'unweighted'}"] = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=15,
            class_weight=weight, random_state=SEED, n_jobs=1,
        )
    return {name: Pipeline([("preprocess", clone(prep)), ("model", model)])
            for name, model in models.items()}


def metrics(y, scores, threshold):
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "alerts": int(fp + tp), "threshold": float(threshold),
    }


def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(
        4 * precision[:-1] + recall[:-1], 1e-15,
    )
    # Resolve ties toward fewer alerts. F2 is a learning objective, not a cost model.
    best = np.flatnonzero(np.isclose(f2, f2.max(), atol=1e-12, rtol=0))[-1]
    return float(thresholds[best])


def bootstrap_intervals(y, scores, threshold, iterations=1000):
    """Stratified bootstrap of this fixed model, conditional on observed prevalence."""
    y, scores = np.asarray(y), np.asarray(scores)
    rng = np.random.default_rng(SEED)
    negatives, positives = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    collected = {key: [] for key in ["average_precision", "precision", "recall"]}
    for _ in range(iterations):
        ix = np.concatenate([rng.choice(negatives, len(negatives), replace=True),
                             rng.choice(positives, len(positives), replace=True)])
        # Faster than computing the complete metric suite on every bootstrap sample.
        guess = scores[ix] >= threshold
        collected["average_precision"].append(average_precision_score(y[ix], scores[ix]))
        collected["precision"].append(precision_score(y[ix], guess, zero_division=0))
        collected["recall"].append(recall_score(y[ix], guess, zero_division=0))
    return {key: [float(v) for v in np.quantile(values, [0.025, 0.975])]
            for key, values in collected.items()}


def train(csv_path: Path, output: Path):
    frame = read_data(csv_path)
    groups = feature_groups(frame)
    y = frame.label.to_numpy()
    train_ix, test_ix = make_splits(y, groups)[0]
    X_train, X_test = frame.iloc[train_ix][FEATURES], frame.iloc[test_ix][FEATURES]
    y_train, y_test = y[train_ix], y[test_ix]
    group_train = groups[train_ix]
    results, oof_by_model = [], {}
    model_options = candidates()
    # All candidates use exactly the same folds. Test labels are never used here.
    repeat_splits = [make_splits(y_train, group_train, SEED + repeat)
                     for repeat in range(3)]
    for name, pipeline in model_options.items():
        repeat_scores, fold_ap = [], []
        for splits in repeat_splits:
            scores = np.full(len(y_train), np.nan)
            for fit, validate in splits:
                fitted = clone(pipeline).fit(X_train.iloc[fit], y_train[fit])
                scores[validate] = fitted.predict_proba(X_train.iloc[validate])[:, 1]
                fold_ap.append(float(average_precision_score(y_train[validate], scores[validate])))
            if not np.isfinite(scores).all():
                raise AssertionError("Every training row must receive a held-out prediction.")
            repeat_scores.append(scores)
        oof = np.mean(repeat_scores, axis=0)
        oof_by_model[name] = oof
        repeat_ap = [float(average_precision_score(y_train, s)) for s in repeat_scores]
        results.append({"model": name, "mean_cv_ap": float(np.mean(repeat_ap)),
                        "cv_ap_by_repeat": repeat_ap,
                        "fold_ap_mean": float(np.mean(fold_ap)),
                        "fold_ap_std": float(np.std(fold_ap)),
                        "fold_ap": fold_ap})
        print(f"{name}: mean validation AP={np.mean(repeat_ap):.4f}", flush=True)
    results.sort(key=lambda row: row["mean_cv_ap"], reverse=True)
    # Include the dummy in selection: beating it is a requirement, not an assumption.
    selected = results[0]["model"]
    threshold = choose_threshold(y_train, oof_by_model[selected])
    fitted = clone(model_options[selected]).fit(X_train, y_train)
    test_scores = fitted.predict_proba(X_test)[:, 1]
    selected_test = metrics(y_test, test_scores, threshold)
    prior_scores = np.full(len(y_test), y_train.mean())
    baseline = metrics(y_test, prior_scores, 0.5)
    always_alert = metrics(y_test, prior_scores, 0.0)
    report = {
        "source": str(csv_path.resolve()),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "seed": SEED, "positive_label": 1, "features": FEATURES,
        "rows": len(frame), "fraud_rows": int(y.sum()),
        "repeated_feature_rows": int(frame.duplicated(FEATURES).sum()),
        "conflicting_feature_groups": int((frame.groupby(FEATURES).label.nunique() > 1).sum()),
        "split": {"training_rows": len(train_ix), "test_rows": len(test_ix),
                  "training_fraud": int(y_train.sum()), "test_fraud": int(y_test.sum()),
                  "feature_group_overlap": len(set(groups[train_ix]) & set(groups[test_ix]))},
        "cross_validation": results, "selected_model": selected,
        "threshold_objective": "Maximize F2 on averaged repeated out-of-fold training scores",
        "threshold": threshold,
        "selected_training_oof": metrics(y_train, oof_by_model[selected], threshold),
        "test": selected_test, "test_at_threshold_0_5": metrics(y_test, test_scores, 0.5),
        "dummy_no_alert_test": baseline, "dummy_always_alert_test": always_alert,
        "test_bootstrap_95_percent_intervals": bootstrap_intervals(y_test, test_scores, threshold),
        "limitations": [
            "Only the supplied CSV was used. Label 1 is assumed to mean fraud.",
            "Random grouped split: no timestamp or customer/account ID is available.",
            "Test results estimate performance on similar rows, not future production traffic.",
            "Model and threshold selection share CV predictions; training CV scores are optimistic.",
            "F2 weights recall more than precision; no business costs were supplied.",
            "Class-weighted scores are not calibrated fraud probabilities.",
            "Bootstrap intervals condition on this model and prevalence; they exclude training uncertainty.",
            "Small positive sample size limits precision of all performance estimates.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": fitted, "threshold": threshold, "features": FEATURES,
                 "selected_model": selected, "source_sha256": report["source_sha256"],
                 "known_categories": {c: sorted(X_train[c].unique().tolist()) for c in CATEGORIES}},
                output / "model.joblib")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(results).drop(columns=["fold_ap", "cv_ap_by_repeat"]).to_csv(
        output / "model_comparison.csv", index=False)
    predictions = frame.iloc[test_ix][["transaction_id", "label"]].copy()
    predictions["fraud_score"] = test_scores
    predictions["predicted_label"] = (test_scores >= threshold).astype(int)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    oof_frame = frame.iloc[train_ix][["transaction_id", "label"]].copy()
    oof_frame["fraud_score"] = oof_by_model[selected]
    oof_frame.to_csv(output / "training_oof_predictions.csv", index=False)
    split_frame = frame[["transaction_id"]].copy()
    split_frame["split"] = "training"
    split_frame.loc[test_ix, "split"] = "test"
    split_frame["feature_group"] = groups
    split_frame.to_csv(output / "split_assignments.csv", index=False)
    (output / "report.md").write_text(render_report(report), encoding="utf-8")
    print(f"Selected {selected}, threshold {threshold:.6f}", flush=True)
    print(json.dumps(selected_test, indent=2), flush=True)
    return report


def render_report(r):
    m, s = r["test"], r["split"]
    rows = "\n".join(
        f"| {v['model']} | {v['mean_cv_ap']:.4f} | {v['fold_ap_std']:.4f} |"
        for v in r["cross_validation"])
    signal = ("The selected model did not exceed the constant-score test AP baseline."
              if m["average_precision"] <= r["dummy_no_alert_test"]["average_precision"]
              else "Test AP exceeds the constant-score reference; this small test alone does not establish reliable predictive value.")
    interval_rows = "\n".join(
        f"| {key.replace('_', ' ').title()} | {low:.4f} | {high:.4f} |"
        for key, (low, high) in r["test_bootstrap_95_percent_intervals"].items())
    return f"""# Fraud classification experiment

Source: `{r['source']}`. Positive label: 1 (assumed fraud).
SHA-256: `{r['source_sha256']}`.

{r['rows']} rows, {r['fraud_rows']} fraud cases. Features: amount, merchant_type, device_type.
Transaction IDs are excluded from model inputs. No external data is used.
There are {r['repeated_feature_rows']} repeated feature rows and {r['conflicting_feature_groups']} conflicting-label feature groups.

## Evaluation design

Fixed seed {r['seed']}. Five-fold stratified group splitting reserves one fold as the test set.
Identical feature profiles remain together; overlapping train/test groups: {s['feature_group_overlap']}.
Training: {s['training_rows']} rows ({s['training_fraud']} fraud).
Test: {s['test_rows']} rows ({s['test_fraud']} fraud).
Three repeats of five-fold stratified group cross-validation select the model by mean repeat average precision (AP).
Each preprocessing pipeline is fitted inside its training fold.
F2 selects the alert threshold using averaged training out-of-fold scores only.
Model and threshold selection scores are optimistic; the reserved test is the final evaluation.

## Training validation comparison

| Candidate | Mean repeat AP | Fold AP standard deviation |
|---|---:|---:|
{rows}

## Final test results

Selected: **{r['selected_model']}**. Alert when score >= **{r['threshold']:.6f}**.
The saved model is trained on the training partition only, matching this evaluation.

| Metric | Result |
|---|---:|
| Average precision | {m['average_precision']:.4f} |
| Constant-score AP reference | {r['dummy_no_alert_test']['average_precision']:.4f} |
| ROC AUC | {m['roc_auc']:.4f} |
| Accuracy | {m['accuracy']:.2%} |
| Balanced accuracy | {m['balanced_accuracy']:.2%} |
| Precision | {m['precision']:.2%} |
| Recall | {m['recall']:.2%} |
| F2 | {m['f2']:.4f} |

| Actual / predicted | Non-fraud | Fraud alert |
|---|---:|---:|
| Non-fraud | {m['true_negatives']} | {m['false_positives']} |
| Fraud | {m['false_negatives']} | {m['true_positives']} |

Fraud caught: {m['true_positives']}. Fraud missed: {m['false_negatives']}. False alarms: {m['false_positives']}.
Always predicting non-fraud gives {r['dummy_no_alert_test']['accuracy']:.2%} accuracy and 0% fraud recall.
Always alerting gives {r['dummy_always_alert_test']['precision']:.2%} precision and 100% recall;
its F2 is {r['dummy_always_alert_test']['f2']:.4f}, versus {m['f2']:.4f} for the selected model.
{signal}

## Uncertainty

Stratified bootstrap 95% intervals (1,000 resamples, fixed model and observed class counts):

| Metric | Lower | Upper |
|---|---:|---:|
{interval_rows}

These intervals exclude training and model-selection uncertainty. One additional detected fraud changes
test recall by {1 / s['test_fraud']:.1%}. Avoid tuning the model against these test results.

## Limitations

""" + "\n".join(f"- {item}" for item in r["limitations"]) + "\n"


def predict(model_path: Path, csv_path: Path, destination: Path):
    # Load only artifacts produced by this project; joblib files can execute code.
    artifact = joblib.load(model_path)
    frame = read_data(csv_path, training=False)
    for column, known in artifact["known_categories"].items():
        unseen = sorted(set(frame[column]) - set(known))
        if unseen:
            print(f"Warning: unseen {column} categories {unseen}; encoded as unknown.")
    scores = artifact["pipeline"].predict_proba(frame[artifact["features"]])[:, 1]
    result = frame.copy()
    result["fraud_score"] = scores
    result["predicted_label"] = (scores >= artifact["threshold"]).astype(int)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Wrote {len(result)} predictions to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--data", type=Path, default=Path("fraud_detection.csv"))
    fit.add_argument("--output", type=Path, default=Path("outputs"))
    score = sub.add_parser("predict")
    score.add_argument("--model", type=Path, default=Path("outputs/model.joblib"))
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output)
    else:
        predict(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
