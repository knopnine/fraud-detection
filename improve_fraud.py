"""Nested evaluation of notebook, previous, and expanded fraud model families."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as SamplingPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from fraud_features import FraudFeatures
from fraud_pipeline import CATEGORIES, FEATURES, candidates as previous_candidates, feature_groups, read_data

SEED = 20261016


def grouped_splits(y, groups, folds, seed):
    result = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups))
    for fit, validate in result:
        if set(groups[fit]) & set(groups[validate]):
            raise AssertionError("A feature group crosses a fold boundary.")
        if len(np.unique(y[fit])) != 2 or len(np.unique(y[validate])) != 2:
            raise ValueError("Both classes are required in every training/validation partition.")
    return result


def engineered_preprocessor(spline=False):
    amount_steps = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]
    if spline:
        amount_steps.append(("splines", SplineTransformer(n_knots=4, degree=2, include_bias=False)))
    amount_steps.append(("scale", StandardScaler()))
    return Pipeline([
        ("features", FraudFeatures()),
        ("encode", ColumnTransformer([
            ("amount", Pipeline(amount_steps), ["amount"]),
            ("numeric", StandardScaler(), ["is_round_amount", "merchant_log_deviation",
                                           "device_log_deviation", "pair_frequency"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORIES + ["amount_category", "merchant_device_combo"]),
        ])),
    ])


def candidate_models():
    models = previous_candidates()
    raw_preprocessor = clone(models["dummy_prior"].named_steps["preprocess"])
    for name, estimator in {
        "notebook_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        "notebook_forest_balanced": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1),
        "notebook_adaboost": AdaBoostClassifier(random_state=42),
    }.items():
        models[name] = Pipeline([("preprocess", clone(raw_preprocessor)), ("model", estimator)])
    for name, weight in [("engineered_logistic", None), ("engineered_logistic_balanced", "balanced")]:
        models[name] = Pipeline([
            ("preprocess", engineered_preprocessor()),
            ("model", LogisticRegression(C=0.1, class_weight=weight, max_iter=2000, random_state=42)),
        ])
    models["spline_logistic_balanced"] = Pipeline([
        ("preprocess", engineered_preprocessor(spline=True)),
        ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    oversample_prep = engineered_preprocessor()
    models["engineered_random_oversampling"] = SamplingPipeline([
        *oversample_prep.steps,
        ("sampler", RandomOverSampler(sampling_strategy=0.25, random_state=42)),
        ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
    ])
    # SMOTENC sees categorical values, with numeric distance measured on scaled log amount.
    mixed_preprocessor = ColumnTransformer([
        ("amount", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), ["amount"]),
        ("categories", "passthrough", CATEGORIES),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    mixed_encode = ColumnTransformer([
        ("amount", "passthrough", ["amount"]),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORIES),
    ])
    models["categorical_smote_logistic"] = SamplingPipeline([
        ("mixed", mixed_preprocessor),
        ("sampler", SMOTENC(categorical_features=CATEGORIES, sampling_strategy=0.25,
                            k_neighbors=3, random_state=42)),
        ("encode", mixed_encode),
        ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
    ])
    for weight in (None, "balanced"):
        models[f"engineered_boost_{weight or 'unweighted'}"] = Pipeline([
            ("preprocess", engineered_preprocessor()),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=100, max_leaf_nodes=7, min_samples_leaf=30,
                l2_regularization=10, class_weight=weight, early_stopping=False, random_state=42)),
        ])
    models["engineered_small_forest"] = Pipeline([
        ("preprocess", engineered_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=20,
                                         class_weight="balanced", n_jobs=1, random_state=42)),
    ])
    return models


def family_names(models):
    return {
        "notebook_representative": ["dummy_prior", "logistic_C1.0_unweighted",
                                    "logistic_C1.0_balanced", "notebook_forest",
                                    "notebook_forest_balanced", "notebook_adaboost"],
        "previous_pipeline": list(previous_candidates()),
        "expanded_approach": list(models),
    }


def select_threshold(y, scores, alert_budget):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    rates = np.asarray([(scores >= value).mean() for value in thresholds])
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-15)
    valid = np.flatnonzero(rates <= alert_budget + 1e-12)
    if not len(valid):
        return float(np.nextafter(np.max(scores), np.inf))
    best = valid[np.flatnonzero(np.isclose(f2[valid], f2[valid].max(), atol=1e-12, rtol=0))[-1]]
    return float(thresholds[best])


def classification_counts(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": precision, "recall": recall,
            "alert_rate": float((tp + fp) / len(y)),
            "accuracy": float((tp + tn) / len(y))}


def evaluate(y, scores, threshold, budget):
    limit = max(1, int(np.ceil(len(y) * budget)))
    # Stable tie handling makes ranking reproducible and avoids using labels for ties.
    top = np.argsort(-np.asarray(scores), kind="stable")[:limit]
    ranked = np.zeros(len(y), dtype=int)
    ranked[top] = 1
    top_metrics = classification_counts(y, ranked)
    return {"ap": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "threshold": threshold, "threshold_metrics": classification_counts(y, scores >= threshold),
            "top_budget_metrics": top_metrics,
            "top_budget_lift": float(top_metrics["precision"] / np.mean(y))}


def inner_evaluate(models, X, y, groups, folds, seed):
    splits = grouped_splits(y, groups, folds, seed)
    results = {}
    for name, estimator in models.items():
        oof, fold_ap = np.full(len(y), np.nan), []
        for fit, validate in splits:
            pipeline = clone(estimator).fit(X.iloc[fit], y[fit])
            scores = pipeline.predict_proba(X.iloc[validate])[:, 1]
            oof[validate] = scores
            fold_ap.append(float(average_precision_score(y[validate], scores)))
        if not np.isfinite(oof).all():
            raise AssertionError("Missing validation scores.")
        results[name] = {"mean_ap": float(np.mean(fold_ap)), "fold_ap": fold_ap, "oof": oof}
    return results


def choose_model(inner, names):
    # Ordered candidate names break exact ties in favor of simpler, earlier candidates.
    return max(names, key=lambda name: inner[name]["mean_ap"])


def permutation_diagnostic(X, y, groups, estimator, count):
    def statistic(labels):
        values = []
        for fit, validate in grouped_splits(labels, groups, 5, SEED + 700):
            p = clone(estimator).fit(X.iloc[fit], labels[fit])
            values.append(average_precision_score(labels[validate], p.predict_proba(X.iloc[validate])[:, 1]))
        return float(np.mean(values))
    observed = statistic(y)
    null, rng = [], np.random.default_rng(SEED + 900)
    for index in range(count):
        null.append(statistic(rng.permutation(y)))
        if (index + 1) % 10 == 0:
            print(f"Permutation diagnostic {index + 1}/{count}", flush=True)
    return {"estimator": "engineered_logistic_balanced", "observed_mean_fold_ap": observed,
            "null_mean_fold_ap": float(np.mean(null)) if null else None,
            "null_95_percent_range": np.quantile(null, [0.025, 0.975]).tolist() if null else [],
            "permutations": count,
            "p_value": float((1 + np.sum(np.asarray(null) >= observed)) / (count + 1)) if count else None,
            "null_scores": null,
            "interpretation": "Fixed prespecified model only; not a selection-adjusted test of the winning search. "
                              "Labels are permuted across rows, preserving total class counts, and splits are rebuilt. "
                              "The test assumes exchangeable transactions; unseen entity/time dependencies cannot be checked."}


def summarize(records, repeats):
    summary = {}
    for family in sorted({v["family"] for v in records}):
        rows = [v for v in records if v["family"] == family]
        top = {k: sum(v["top_budget_metrics"][k] for v in rows) / repeats for k in ["tp", "fp", "fn", "tn"]}
        decisions = {k: sum(v["threshold_metrics"][k] for v in rows) / repeats for k in ["tp", "fp", "fn", "tn"]}
        summary[family] = {
            "mean_outer_ap": float(np.mean([v["ap"] for v in rows])),
            "outer_ap_std": float(np.std([v["ap"] for v in rows])),
            "mean_outer_roc_auc": float(np.mean([v["roc_auc"] for v in rows])),
            "mean_top_budget_precision": float(top["tp"] / (top["tp"] + top["fp"])),
            "mean_top_budget_recall": float(top["tp"] / (top["tp"] + top["fn"])),
            "top_budget_counts_per_repeat": top,
            "threshold_counts_per_repeat": decisions,
            "threshold_precision": float(decisions["tp"] / (decisions["tp"] + decisions["fp"])) if decisions["tp"] + decisions["fp"] else 0,
            "threshold_recall": float(decisions["tp"] / (decisions["tp"] + decisions["fn"])),
            "selected_models": dict(Counter(v["selected_model"] for v in rows)),
            "per_repeat_mean_ap": [float(np.mean([v["ap"] for v in rows if v["repeat"] == repeat]))
                                   for repeat in range(repeats)],
        }
    return summary


def write_report(report, output):
    lines = ["# Improved fraud experiment", "", report["assessment"], "",
             "All results below come from the same outer folds. This is a new exploratory analysis",
             "of an already inspected dataset, not validation on newly collected data.", "",
             "## Nested comparison", "",
             "| Selection procedure | Mean fold AP | Mean fold ROC AUC | Precision at top 10% | Recall at top 10% |",
             "|---|---:|---:|---:|---:|"]
    for name, values in report["summary"].items():
        lines.append(f"| {name} | {values['mean_outer_ap']:.4f} | {values['mean_outer_roc_auc']:.4f} | "
                     f"{values['mean_top_budget_precision']:.2%} | {values['mean_top_budget_recall']:.2%} |")
    lines += ["", "AP means average precision. The constant-score AP baseline is the fold fraud fraction, approximately 0.05.",
              "Finite-sample random ranking can have mean AP above this constant-score reference; the shuffled-label diagnostic gives additional context.",
              "The exact top-budget metric ranks each evaluation batch and alerts on the top 10% (ties use row order).",
              "This is a capacity policy, not a calibrated probability threshold. It assumes the whole batch can be scored together.",
              "", "## Threshold behavior", "",
              "Every family chooses a threshold using inner validation scores only, maximizing F2 while alerting on at most 10% of those scores.",
              "Realized outer alert rates may exceed 10% as score distributions shift. Exact top-budget ranking is reported separately.",
              "Counts below are averaged over repeats, so every row contributes once per repeat; these are not independent new transactions.", "",
              "| Procedure | Fraud caught / 50 | False alarms | Precision | Recall |",
              "|---|---:|---:|---:|---:|"]
    for name, values in report["summary"].items():
        counts = values["threshold_counts_per_repeat"]
        lines.append(f"| {name} | {counts['tp']:.1f} | {counts['fp']:.1f} | "
                     f"{values['threshold_precision']:.2%} | {values['threshold_recall']:.2%} |")
    p = report["permutation_diagnostic"]
    lines += ["", "## Signal diagnostic", "",
              f"Prespecified engineered logistic regression: observed mean-fold AP {p['observed_mean_fold_ap']:.4f}.",
              f"Shuffled-label mean-fold AP: {p['null_mean_fold_ap']}. Permutation p-value: {p['p_value']} ({p['permutations']} permutations).",
              p["interpretation"], "", "## Method", "",
              f"- {report['plan']['outer_repeats']} repeats of {report['plan']['outer_folds']}-fold stratified group outer validation.",
              f"- {report['plan']['inner_folds']}-fold inner validation selects candidates by mean fold AP, and chooses the decision threshold.",
              "- Exact feature profiles stay together at both levels. IDs are excluded. All transformations and sampling fit inside training folds.",
              "- Merchant/device log-amount means and pair frequencies are learned in FraudFeatures.fit; unknown categories use training fallbacks.",
              "- Same outer rows, folds, seeds, threshold objective and batch alert budget for every selection procedure.",
              "- Sampling ratio 0.25 means one minority example per four majority examples after resampling. Class weights are not combined with oversampling.",
              "- The candidate list, seed and evaluation policy were saved before evaluating outer-fold results.",
              "- Repeated-fold standard deviations describe variability; folds/repeats overlap in training data and are not independent confidence intervals.",
              "- Historical notebook tuning and earlier test inspection cannot be undone. Nested validation reduces new selection bias but is not a fresh external test.",
              "", "## Saved learning model", "",
              f"Full-data inner-validation choice: `{report['final_model']['name']}`.",
              f"Threshold: {report['final_model']['threshold']:.6f}. Scores are not calibrated probabilities.",
              "The saved model is fitted on all 1,000 rows for further learning/inference. Its training scores are not a performance estimate.",
              "Outer results evaluate the selection procedure, not this exact full-data fit. No production accuracy claim is made.",
              "", "## Files", "",
              "- `experiment_plan.json`: candidate names and evaluation policy frozen before the run.",
              "- `report.json`: full methodology, fold metrics, selected models, and diagnostic.",
              "- `outer_predictions.csv`: per-row outer predictions, fold, repeat, thresholds, and group IDs.",
              "- `candidate_inner_scores.csv`: candidate selection scores within each outer fold.",
              "- `model.joblib`: full-data fitted pipeline, threshold and metadata.",
              "", "## References", "",
              "- [Nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)",
              "- [Categorical-aware sampling](https://imbalanced-learn.org/stable/over_sampling.html)"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_path, output, repeats=2, permutations=99):
    X_frame = read_data(data_path)
    X, y = X_frame[FEATURES], X_frame.label.to_numpy()
    groups = feature_groups(X_frame)
    models, records, prediction_rows, selection_rows = candidate_models(), [], [], []
    families = family_names(models)
    plan = {"seed": SEED, "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "outer_folds": 5, "outer_repeats": repeats, "inner_folds": 3, "alert_budget": 0.10,
            "candidates": list(models), "families": families, "selection_metric": "mean inner fold average precision",
            "permutation_model": "engineered_logistic_balanced", "permutations": permutations,
            "interpretation": "Exploratory nested comparison on already inspected data; no untouched external test."}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "experiment_plan.json"
    if plan_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {output}. Choose a new output directory.")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    for repeat in range(repeats):
        outer = grouped_splits(y, groups, 5, SEED + repeat)
        for fold, (fit, validate) in enumerate(outer):
            print(f"Outer repeat {repeat + 1}/{repeats}, fold {fold + 1}/5: comparing {len(models)} candidates", flush=True)
            inner = inner_evaluate(models, X.iloc[fit], y[fit], groups[fit], 3, SEED + repeat * 10 + fold + 100)
            selection_rows.extend({"repeat": repeat, "fold": fold, "model": name, "inner_ap": v["mean_ap"]}
                                  for name, v in inner.items())
            fitted_cache = {}
            for family, names in families.items():
                name = choose_model(inner, names)
                threshold = select_threshold(y[fit], inner[name]["oof"], 0.10)
                if name not in fitted_cache:
                    fitted_cache[name] = clone(models[name]).fit(X.iloc[fit], y[fit])
                scores = fitted_cache[name].predict_proba(X.iloc[validate])[:, 1]
                result = evaluate(y[validate], scores, threshold, 0.10)
                record = {"family": family, "repeat": repeat, "fold": fold, "selected_model": name,
                          "test_rows": len(validate), "test_fraud": int(y[validate].sum()), **result}
                records.append(record)
                top_indices = np.argsort(-scores, kind="stable")[:int(np.ceil(len(scores) * 0.10))]
                top_flags = np.zeros(len(scores), dtype=int)
                top_flags[top_indices] = 1
                prediction_rows.extend({"family": family, "repeat": repeat, "fold": fold,
                                        "transaction_id": int(X_frame.iloc[ix].transaction_id),
                                        "group": int(groups[ix]), "label": int(y[ix]),
                                        "fraud_score": float(scores[j]), "threshold": threshold,
                                        "threshold_alert": int(scores[j] >= threshold), "top_budget_alert": int(top_flags[j]),
                                        "selected_model": name}
                                       for j, ix in enumerate(validate))
                print(f"  {family}: {name}; outer AP={result['ap']:.4f}; top-budget fraud={result['top_budget_metrics']['tp']}", flush=True)
            # Checkpoints preserve completed work without changing the experiment plan.
            pd.DataFrame(prediction_rows).to_csv(output / "outer_predictions.csv", index=False)
            pd.DataFrame(selection_rows).to_csv(output / "candidate_inner_scores.csv", index=False)
            (output / "fold_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Running prespecified shuffled-label diagnostic", flush=True)
    diagnostic = permutation_diagnostic(X, y, groups, models["engineered_logistic_balanced"], permutations)
    print("Selecting a full-data learning model using inner validation", flush=True)
    full_inner = inner_evaluate(models, X, y, groups, 3, SEED + 500)
    final_name = choose_model(full_inner, families["expanded_approach"])
    final_threshold = select_threshold(y, full_inner[final_name]["oof"], 0.10)
    fitted = clone(models[final_name]).fit(X, y)
    joblib.dump({"pipeline": fitted, "threshold": final_threshold, "features": FEATURES,
                 "selected_model": final_name, "source_sha256": plan["source_sha256"],
                 "known_categories": {c: sorted(X[c].unique().tolist()) for c in CATEGORIES},
                 "purpose": "Full-data learning fit; nested results describe selection procedure."}, output / "model.joblib")
    summary = summarize(records, repeats)
    improved = summary["expanded_approach"]["mean_outer_ap"] > max(
        summary["previous_pipeline"]["mean_outer_ap"], summary["notebook_representative"]["mean_outer_ap"])
    assessment = ("The expanded procedure has higher mean outer AP in this experiment; this is exploratory evidence, not confirmed improvement on new data."
                  if improved else "The expanded procedure did not beat both comparison procedures in mean outer AP. No consistent model improvement was established.")
    expanded = summary["expanded_approach"]
    if expanded["mean_top_budget_precision"] <= y.mean() + 1e-12:
        assessment += " At the fixed review budget, expanded precision did not exceed the dataset's fraud prevalence; no useful advantage over chance was established at that workload."
    report = {"plan": plan, "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
               "imbalanced_learn": imblearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
              "assessment": assessment, "summary": summary, "fold_results": records,
              "permutation_diagnostic": diagnostic,
              "final_model": {"name": final_name, "threshold": final_threshold,
                              "training_rows": len(X), "training_fraud": int(y.sum()),
                              "full_data_inner_scores": {name: v["mean_ap"] for name, v in full_inner.items()}}}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report, output)
    print(assessment, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("fraud_detection.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/improved_v1"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=99)
    args = parser.parse_args()
    if args.repeats < 1 or args.permutations < 0:
        parser.error("repeats must be positive and permutations nonnegative")
    with threadpool_limits(limits=1):
        run(args.data, args.output, args.repeats, args.permutations)


if __name__ == "__main__":
    main()
