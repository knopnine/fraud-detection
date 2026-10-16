"""Build and execute the lightweight learning notebook from saved experiment results."""

import contextlib
import io
import json
import time
from pathlib import Path


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown("""# Fraud classification: improving the experiment

This notebook builds on your original GitHub project using **only the same 1,000-row CSV**.
It separates model and threshold selection from evaluation with nested validation.
It compares representative notebook models, our previous pipeline, and an expanded family
using the same outer folds. Results below are from the completed experiment.

Label `1` is assumed to mean fraud. The dataset has already been inspected and used in
earlier experiments. This is an exploratory comparison, not an untouched external test.
Improved methodology does not guarantee improved predictions.

Use the project's `.venv` as the notebook kernel. All feature and training logic is in
`fraud_features.py` and `improve_fraud.py`, keeping this notebook focused on the learning process.
"""),
    code("""from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from fraud_pipeline import FEATURES, read_data, feature_groups
from fraud_features import FraudFeatures
from improve_fraud import candidate_models, family_names, grouped_splits, run

DATA_PATH = Path('fraud_detection.csv')
EXPERIMENT_DIR = Path('reports/improved_v1')
df = read_data(DATA_PATH)
print(f'Python: {sys.version.split()[0]}')
print(f'Rows: {len(df)}, fraud: {df.label.sum()}, fraud fraction: {df.label.mean():.1%}')
print(df.groupby('label').amount.agg(['count', 'mean', 'median']).to_string())
"""),
    markdown("""## 1. Establish what success means

Always predicting non-fraud achieves 95% accuracy and zero fraud recall. A useful model
must rank fraud above non-fraud, or catch more fraud for the same review workload.

We use **average precision (AP)** for model selection. ROC AUC, precision, recall, and
confusion counts provide context. The constant-score AP reference is 0.05. Random rankings
on small samples may score above 0.05, so a shuffled-label diagnostic is also included.

For a workload comparison, we rank each outer evaluation batch and inspect its top 10%.
The 10% budget is an illustrative assumption, not a business requirement supplied by the data.
At chance, reviewing 100 of these 1,000 transactions would catch about five of the 50 fraud cases.
"""),
    code("""baseline_accuracy = (df.label == 0).mean()
print(f'Always non-fraud accuracy: {baseline_accuracy:.1%}, recall: 0%')
print(f'Constant-score AP reference: {df.label.mean():.4f}')
print('Chance expectation at 10% review volume: about 5 fraud / 100 reviews across 1,000 rows.')
"""),
    markdown("""## 2. Add features without using evaluation rows

The original notebook calculated merchant means over the complete CSV. Here the transformer
learns merchant/device mean log amounts and merchant/device frequencies only in `fit`.
`transform` uses those stored statistics, including training-derived fallbacks for unknown categories.
Amount bands, round-number flags, and category combinations depend only on the current row.

The demonstration below fits one training partition. During the actual experiment, the same
transformer is fitted again inside every inner and outer training fold.
"""),
    code("""X = df[FEATURES]
y = df.label.to_numpy()
groups = feature_groups(df)
fit_ix, validation_ix = grouped_splits(y, groups, 5, 20261016)[0]
features = FraudFeatures().fit(X.iloc[fit_ix])
example = features.transform(X.iloc[validation_ix[:4]])
print(example.to_string(index=False))
assert not set(groups[fit_ix]) & set(groups[validation_ix])
print('No exact feature profiles overlap between these partitions.')
"""),
    markdown("""## 3. Freeze the candidate families before outer evaluation

The expanded family includes regularized logistic regression, nonlinear amount splines,
your amount/category interaction ideas, constrained forests, and histogram gradient boosting.
It also compares random oversampling and SMOTENC with no resampling/class weighting.
SMOTENC sees categorical values before one-hot encoding; numeric neighbor distances use scaled log amounts.
Sampling occurs inside training folds. Oversampling and class weighting are not combined.

The notebook comparison is a **representative subset** of your notebook's models, not a
reproduction of all boosting libraries and every historical hyperparameter search.
"""),
    code("""models = candidate_models()
families = family_names(models)
for family, names in families.items():
    print(f'{family}: {len(names)} candidates')
print('\\nExpanded candidates:')
print('\\n'.join(models))
"""),
    markdown("""## 4. Run nested validation, or use the saved completed run

There are two repeats of five outer folds. Within each outer training partition, three inner
folds select the model using mean AP and select an alert threshold using F2, restricted to at
most 10% of inner out-of-fold scores. Each selected pipeline then scores its outer evaluation rows.

The experiment plan is saved before outer results are computed. The script refuses to overwrite
an existing run. Set `RUN_EXPERIMENT = True` to create a new timestamped run. Repeating or
changing the experiment after seeing its results does not create new independent evidence.
"""),
    code("""RUN_EXPERIMENT = False
if RUN_EXPERIMENT:
    from datetime import datetime
    EXPERIMENT_DIR = Path('outputs') / ('improved_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    with threadpool_limits(limits=1):
        run(DATA_PATH, EXPERIMENT_DIR, repeats=2, permutations=99)

report = json.loads((EXPERIMENT_DIR / 'report.json').read_text(encoding='utf-8'))
print(report['assessment'])
print('Source SHA-256:', report['plan']['source_sha256'])
"""),
    markdown("""## 5. Compare the selection procedures on identical outer folds

These metrics describe each **model-selection procedure**. A different candidate can be selected
in each outer fold. Fold standard deviations are descriptive because training samples overlap.
Do not treat the ten folds as ten independent experiments or count repeated fraud cases as new cases.
"""),
    code("""rows = []
for family, result in report['summary'].items():
    rows.append({'procedure': family, 'mean_fold_AP': result['mean_outer_ap'],
                 'mean_fold_ROC_AUC': result['mean_outer_roc_auc'],
                 'top_10pct_precision': result['mean_top_budget_precision'],
                 'top_10pct_recall': result['mean_top_budget_recall']})
comparison = pd.DataFrame(rows).set_index('procedure')
print(comparison.round(4).to_string())
print('\\nChoices within the expanded procedure:')
print(report['summary']['expanded_approach']['selected_models'])
print('\\nMean fold AP in each repeat:')
for name, result in report['summary'].items():
    print(name, [round(v, 4) for v in result['per_repeat_mean_ap']])
"""),
    markdown("""## 6. Compare alert counts and threshold behavior

The saved decision threshold is selected without the outer evaluation labels. Its realized alert
volume can change as score distributions shift. This differs from selecting exactly the top 10%
of each evaluation batch. Ranking a batch requires access to the batch before choosing alerts.
Counts below are averaged over repeats, with each of the 1,000 rows scored once per repeat.
"""),
    code("""for name, result in report['summary'].items():
    print('\\n' + name)
    print('Top 10% counts per repeat:', result['top_budget_counts_per_repeat'])
    print('Learned-threshold counts per repeat:', result['threshold_counts_per_repeat'])
    print(f\"Threshold precision={result['threshold_precision']:.2%}, recall={result['threshold_recall']:.2%}\")
"""),
    markdown("""## 7. Check whether a prespecified model beats shuffled labels

The diagnostic evaluates one prespecified engineered logistic model with real labels and 99
label permutations. All learned preprocessing is refitted in each validation fold. A large p-value
means this diagnostic did not distinguish the real-label score from shuffled-label scores.
It does not prove that all possible classifiers fail, and it is not a selection-adjusted test of
the best model from the complete search. It assumes transaction labels are exchangeable;
customer and time dependence cannot be checked with the available columns.
"""),
    code("""diagnostic = report['permutation_diagnostic']
for key in ['estimator', 'observed_mean_fold_ap', 'null_mean_fold_ap',
            'null_95_percent_range', 'permutations', 'p_value']:
    print(f'{key}: {diagnostic[key]}')
"""),
    markdown("""## 8. Inspect the full-data learning model

After the comparison is complete, inner validation on all rows chooses a full-data model.
That fitted artifact can accept the original three predictors. The examples below illustrate
inference on existing rows; they are **not held-out performance measurements**.
The outer-fold metrics describe the training/selection procedure rather than this exact fit.
Scores are not calibrated fraud probabilities. Load only trusted joblib files.
"""),
    code("""import joblib
print('Selected full-data model:', report['final_model']['name'])
print('Threshold:', report['final_model']['threshold'])
model_path = EXPERIMENT_DIR / 'model.joblib'
if model_path.exists():
    artifact = joblib.load(model_path)
    with threadpool_limits(limits=1):
        scores = artifact['pipeline'].predict_proba(X.head(3))[:, 1]
    example_predictions = df.head(3)[['transaction_id']].copy()
    example_predictions['fraud_score'] = scores
    example_predictions['alert'] = (scores >= artifact['threshold']).astype(int)
    print(example_predictions.to_string(index=False))
else:
    print('Model binaries are generated locally. Set RUN_EXPERIMENT = True to train and inspect one.')
"""),
    markdown("""## What this experiment can teach us

Read `reports/improved_v1/report.md` for the completed comparison and limitations. Prefer a
consistent improvement across evaluation folds and alert workloads over the highest isolated score.
The outer fold predictions and all inner selection scores are saved so the results can be audited.
If the added features and models do not improve ranking or fraud caught at a fixed workload,
report that finding instead of selecting a lucky split. Additional independent evidence would
require genuinely new labeled transactions, rather than another split of this already inspected CSV.
"""),
]


def main():
    namespace, execution = {"__name__": "__notebook__"}, 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        execution += 1
        output, start = io.StringIO(), time.perf_counter()
        with contextlib.redirect_stdout(output):
            exec(compile("".join(cell["source"]), f"notebook_cell_{execution}", "exec"), namespace)
        cell["execution_count"] = execution
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output.getvalue().splitlines(keepends=True)}]
        print(f"Executed notebook cell {execution} ({time.perf_counter() - start:.2f}s)")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python (fraud project .venv)",
                                             "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": namespace['sys'].version.split()[0]}},
                "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"fraud-learning-{index:02d}"
    Path("improved_notebook.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Saved improved_notebook.ipynb with executed outputs")


if __name__ == "__main__":
    main()
