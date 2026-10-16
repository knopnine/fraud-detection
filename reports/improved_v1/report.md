# Improved fraud experiment

The expanded procedure has higher mean outer AP in this experiment; this is exploratory evidence, not confirmed improvement on new data. At the fixed review budget, expanded precision did not exceed the dataset's fraud prevalence; no useful advantage over chance was established at that workload.

All results below come from the same outer folds. This is a new exploratory analysis
of an already inspected dataset, not validation on newly collected data.

## Nested comparison

| Selection procedure | Mean fold AP | Mean fold ROC AUC | Precision at top 10% | Recall at top 10% |
|---|---:|---:|---:|---:|
| expanded_approach | 0.0602 | 0.4959 | 5.00% | 10.00% |
| notebook_representative | 0.0586 | 0.5057 | 4.50% | 9.00% |
| previous_pipeline | 0.0560 | 0.4544 | 3.50% | 7.00% |

AP means average precision. The constant-score AP baseline is the fold fraud fraction, approximately 0.05.
Finite-sample random ranking can have mean AP above this constant-score reference; the shuffled-label diagnostic gives additional context.
The exact top-budget metric ranks each evaluation batch and alerts on the top 10% (ties use row order).
This is a capacity policy, not a calibrated probability threshold. It assumes the whole batch can be scored together.

## Threshold behavior

Every family chooses a threshold using inner validation scores only, maximizing F2 while alerting on at most 10% of those scores.
Realized outer alert rates may exceed 10% as score distributions shift. Exact top-budget ranking is reported separately.
Counts below are averaged over repeats, so every row contributes once per repeat; these are not independent new transactions.

| Procedure | Fraud caught / 50 | False alarms | Precision | Recall |
|---|---:|---:|---:|---:|
| expanded_approach | 0.5 | 56.0 | 0.88% | 1.00% |
| notebook_representative | 0.0 | 18.0 | 0.00% | 0.00% |
| previous_pipeline | 1.0 | 49.0 | 2.00% | 2.00% |

## Signal diagnostic

Prespecified engineered logistic regression: observed mean-fold AP 0.0705.
Shuffled-label mean-fold AP: 0.07285535251475464. Permutation p-value: 0.51 (99 permutations).
Fixed prespecified model only; not a selection-adjusted test of the winning search. Labels are permuted across rows, preserving total class counts, and splits are rebuilt. The test assumes exchangeable transactions; unseen entity/time dependencies cannot be checked.

## Method

- 2 repeats of 5-fold stratified group outer validation.
- 3-fold inner validation selects candidates by mean fold AP, and chooses the decision threshold.
- Exact feature profiles stay together at both levels. IDs are excluded. All transformations and sampling fit inside training folds.
- Merchant/device log-amount means and pair frequencies are learned in FraudFeatures.fit; unknown categories use training fallbacks.
- Same outer rows, folds, seeds, threshold objective and batch alert budget for every selection procedure.
- Sampling ratio 0.25 means one minority example per four majority examples after resampling. Class weights are not combined with oversampling.
- The candidate list, seed and evaluation policy were saved before evaluating outer-fold results.
- Repeated-fold standard deviations describe variability; folds/repeats overlap in training data and are not independent confidence intervals.
- Historical notebook tuning and earlier test inspection cannot be undone. Nested validation reduces new selection bias but is not a fresh external test.

## Saved learning model

Full-data inner-validation choice: `engineered_small_forest`.
Threshold: 0.578679. Scores are not calibrated probabilities.
The saved model is fitted on all 1,000 rows for further learning/inference. Its training scores are not a performance estimate.
Outer results evaluate the selection procedure, not this exact full-data fit. No production accuracy claim is made.

## Files

- `experiment_plan.json`: candidate names and evaluation policy frozen before the run.
- `report.json`: full methodology, fold metrics, selected models, and diagnostic.
- `outer_predictions.csv`: per-row outer predictions, fold, repeat, thresholds, and group IDs.
- `candidate_inner_scores.csv`: candidate selection scores within each outer fold.
- `model.joblib`: full-data fitted pipeline, threshold and metadata.

## References

- [Nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Categorical-aware sampling](https://imbalanced-learn.org/stable/over_sampling.html)
