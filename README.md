# Fraud Detection Project

A learning project exploring fraud classification using the supplied
[`fraud_detection.csv`](fraud_detection.csv). The original
[`notebook.ipynb`](notebook.ipynb) explores multiple classifiers, imbalance
strategies, feature engineering, hyperparameter tuning, and MLflow tracking.

The added [`improved_notebook.ipynb`](improved_notebook.ipynb) contains an executed,
reproducible comparison with leakage-safe feature engineering and nested
cross-validation. It uses the same dataset and no external transaction data.

## Experiment report

The dataset contains **1,000 transactions**, including **50 fraud labels** and
950 non-fraud labels. Label `1` is treated as fraud. Inputs are `amount`,
`merchant_type`, and `device_type`; `transaction_id` is excluded from predictors.

The expanded experiment compared **18 configurations**. All three selection
procedures below use identical outer evaluation folds, so their results are
comparable within this experiment.

| Selection procedure | Mean outer-fold average precision | Mean outer-fold ROC AUC | Precision at 10% review volume | Recall at 10% review volume |
|---|---:|---:|---:|---:|
| Representative original-notebook models | 0.0586 | 0.5057 | 4.5% | 9% |
| Initial reproducible pipeline | 0.0560 | 0.4544 | 3.5% | 7% |
| Expanded approach | **0.0602** | 0.4959 | **5.0%** | **10%** |

The expanded approach has a small numerical increase in average precision.
At the fixed review volume it catches an average of **5 of 50 fraud cases**,
with **95 false alarms**, per repeat. This matches the expectation from randomly
reviewing 10% of these transactions. The fixed engineered-logistic diagnostic
gave a shuffled-label permutation **p-value of 0.51**, providing no clear evidence
of useful predictive signal for that prespecified model. It is not a
selection-adjusted test of the winning model from the full search.

**These results do not establish a reliable fraud classifier.** Amounts overlap
between classes, and the available merchant/device categories provide little
consistent separation in the tested models. Only 50 fraud examples are available.
The experiments suggest limited predictive information in these columns; they
do not prove that every possible model would fail.

Always predicting non-fraud achieves **95% accuracy while detecting zero fraud**.
Accuracy alone is therefore not a useful success criterion for this project.

Detailed results and audit files:

- [Expanded experiment report](reports/improved_v1/report.md), including threshold behavior and limitations.
- [Machine-readable results](reports/improved_v1/report.json).
- [Frozen experiment plan](reports/improved_v1/experiment_plan.json).
- [Outer-fold predictions](reports/improved_v1/outer_predictions.csv) and [inner candidate scores](reports/improved_v1/candidate_inner_scores.csv).
- [Initial holdout experiment](reports/original_holdout/report.md), evaluated on a different split and not directly comparable with the nested table above.

## What changed in the approach

- Merchant/device mean log amounts and pair frequencies are learned inside
  training folds. The original notebook calculated merchant means before splitting.
- Feature engineering adds fixed amount bands, round-number flags,
  merchant/device interactions, and deviations from training reference means.
- The candidate family includes regularized logistic regression, amount splines,
  constrained forests, histogram gradient boosting, random oversampling, and
  categorical-aware SMOTE. Oversampling is not combined with class weighting.
- All preprocessing and sampling occur inside training folds. SMOTENC handles
  categories before one-hot encoding; it does not interpolate one-hot indicators.
- Two repeats of five-fold outer validation evaluate the complete selection
  procedure. Three inner folds select the model by mean average precision and
  choose a threshold using F2 with a 10% inner-validation alert limit.
- Identical predictor profiles remain together in both inner and outer splits.
- The top-10% batch-ranking policy is evaluated separately from learned thresholds.
  A learned threshold's realized alert volume can change between datasets.

The original-notebook comparison is a representative subset: logistic regression,
random forest, AdaBoost, weighting where applicable, and a dummy baseline. It does
not reproduce every historical boosting-library and resampling search.

## Reproduce the experiment

Run commands from the repository root. The completed experiment used Python
3.14.0; exact installed versions are in
[`requirements-lock.txt`](requirements-lock.txt). These requirements cover the
new standalone experiment. The original notebook's broader environment remains
defined by `pyproject.toml` and `uv.lock`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe improve_fraud.py --output outputs/improved_rerun
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.
For compatible dependency ranges instead of the exact run environment, install
`requirements.txt`. Reproduction across library versions may produce different
results.

The default experiment uses 99 label permutations and can take several minutes.
An existing experiment directory is protected against overwriting. Each run saves
its plan, dataset SHA-256 fingerprint, versions, fold metrics, predictions,
candidate scores, report, and a full-data fitted learning model.

The saved notebook includes executed outputs and reads the tracked results in
`reports/improved_v1`. To run it interactively, use a Jupyter kernel with the
experiment dependencies installed. If needed:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyter ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --user --name fraud-detection --display-name "Fraud Detection"
.\.venv\Scripts\python.exe -m jupyter notebook improved_notebook.ipynb
```

Select **Fraud Detection** as the kernel. Setting `RUN_EXPERIMENT = True` in the
notebook creates a new timestamped run. The full-data model is generated locally;
model binaries and output directories are ignored by Git.

## Use a generated model

After running the experiment:

```powershell
.\.venv\Scripts\python.exe fraud_pipeline.py predict --model outputs/improved_rerun/model.joblib --data path/to/transactions.csv --output outputs/predictions.csv
```

Prediction input requires `amount`, `merchant_type`, and `device_type`. IDs and
labels are optional. Output includes a ranking score and a threshold-based alert.
Scores are not calibrated fraud probabilities. The final learning model is fitted
on all 1,000 rows after evaluation; predictions on those same rows are not a
held-out performance estimate. Only load trusted joblib artifacts.

## Validation and limitations

Eight automated tests cover invalid input, missing fraud hidden by high accuracy,
threshold selection, nested group separation, training-only feature statistics,
unseen categories, and fitting every candidate. The completed run's 6,000 outer
prediction records were independently checked against the source labels and
reported metrics. These include repeated predictions for the same 1,000 rows,
not 6,000 independent transactions.

The CSV was already inspected and used for earlier experiments. Nested validation
reduces bias from the new model search but does not create a fresh external test.
Fold results overlap in training data, so their spread is not an independent
confidence interval. No timestamp, customer ID, account history, or transaction
sequence is available to assess temporal or customer-level generalization.

This dataset remains useful for learning the classification workflow and the
limits of modeling. Stronger claims would require genuinely new labeled evidence.

## Method references

- [Scikit-learn: nested versus non-nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- [Scikit-learn: preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [Imbalanced-learn: categorical-aware sampling](https://imbalanced-learn.org/stable/over_sampling.html).
