# Fraud classification experiment

Source: `fraud_detection.csv`. Positive label: 1 (assumed fraud).
SHA-256: `8774d2f73aff78e4db3b44dc8ae7e52f00db8be6ea77374fd2e64c64acc6b668`.

1000 rows, 50 fraud cases. Features: amount, merchant_type, device_type.
Transaction IDs are excluded from model inputs. No external data is used.
There are 2 repeated feature rows and 0 conflicting-label feature groups.

## Evaluation design

Fixed seed 42. Five-fold stratified group splitting reserves one fold as the test set.
Identical feature profiles remain together; overlapping train/test groups: 0.
Training: 800 rows (40 fraud).
Test: 200 rows (10 fraud).
Three repeats of five-fold stratified group cross-validation select the model by mean repeat average precision (AP).
Each preprocessing pipeline is fitted inside its training fold.
F2 selects the alert threshold using averaged training out-of-fold scores only.
Model and threshold selection scores are optimistic; the reserved test is the final evaluation.

## Training validation comparison

| Candidate | Mean repeat AP | Fold AP standard deviation |
|---|---:|---:|
| forest_balanced | 0.0671 | 0.0419 |
| forest_unweighted | 0.0603 | 0.0301 |
| logistic_C1.0_balanced | 0.0561 | 0.0292 |
| logistic_C0.1_balanced | 0.0558 | 0.0280 |
| logistic_C1.0_unweighted | 0.0553 | 0.0236 |
| logistic_C0.1_unweighted | 0.0547 | 0.0220 |
| dummy_prior | 0.0500 | 0.0000 |

## Final test results

Selected: **forest_balanced**. Alert when score >= **0.528784**.
The saved model is trained on the training partition only, matching this evaluation.

| Metric | Result |
|---|---:|
| Average precision | 0.0553 |
| Constant-score AP reference | 0.0500 |
| ROC AUC | 0.4947 |
| Accuracy | 77.50% |
| Balanced accuracy | 45.53% |
| Precision | 2.70% |
| Recall | 10.00% |
| F2 | 0.0649 |

| Actual / predicted | Non-fraud | Fraud alert |
|---|---:|---:|
| Non-fraud | 154 | 36 |
| Fraud | 9 | 1 |

Fraud caught: 1. Fraud missed: 9. False alarms: 36.
Always predicting non-fraud gives 95.00% accuracy and 0% fraud recall.
Always alerting gives 5.00% precision and 100% recall;
its F2 is 0.2083, versus 0.0649 for the selected model.
Test AP exceeds the constant-score reference; this small test alone does not establish reliable predictive value.

## Uncertainty

Stratified bootstrap 95% intervals (1,000 resamples, fixed model and observed class counts):

| Metric | Lower | Upper |
|---|---:|---:|
| Average Precision | 0.0417 | 0.0991 |
| Precision | 0.0000 | 0.0857 |
| Recall | 0.0000 | 0.3000 |

These intervals exclude training and model-selection uncertainty. One additional detected fraud changes
test recall by 10.0%. Avoid tuning the model against these test results.

## Limitations

- Only the supplied CSV was used. Label 1 is assumed to mean fraud.
- Random grouped split: no timestamp or customer/account ID is available.
- Test results estimate performance on similar rows, not future production traffic.
- Model and threshold selection share CV predictions; training CV scores are optimistic.
- F2 weights recall more than precision; no business costs were supplied.
- Class-weighted scores are not calibrated fraud probabilities.
- Bootstrap intervals condition on this model and prevalence; they exclude training uncertainty.
- Small positive sample size limits precision of all performance estimates.
