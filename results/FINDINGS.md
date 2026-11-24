# Findings: 30-day readmission of diabetic patients

Data: Diabetes 130-US Hospitals 1999-2008 (UCI repository, CC BY 4.0; Strack et al., 2014).
All numbers below are generated from `results/metrics.json` by `scripts/write_findings.py`.

## Summary
- After cleaning, **69,987 patients** remain; **9.0%** were readmitted within 30 days.
- The best model by cross-validation is **gradient boosting**: test ROC-AUC **0.654** (95% bootstrap CI 0.639-0.670), PR-AUC 0.186 against a 0.090 baseline.
- Logistic regression is close behind (test ROC-AUC 0.648), so the simpler, explainable model gives up only 0.006 AUC.
- Flagging the riskiest 10% of patients catches **24%** of readmissions, with 21.9% of flagged patients readmitted (2.4x the average).
- Readmission is only modestly predictable from these administrative records: useful for ranking who gets a follow-up call, not for deciding individual care.

## What drives the risk
- **Prior inpatient visits:** 8.1% readmitted with none in the past year vs 29.3% with 4 or more.
- **Discharge destination:** the highest rate among groups of 300+ patients is 'Rehab or other hospital' (20.6%, n=3,997).
- **HbA1c** was measured for only 18.4% of patients (readmitted 8.4% vs 9.1% when not measured).

Largest logistic-regression effects (odds ratio; per SD for numbers, per category vs the average category):

| Feature | Level | Odds ratio |
|---|---|---|
| discharge_disposition_id | 22: Discharged/transferred to another rehab fac including rehab units of a hospital . | 2.47 |
| discharge_disposition_id | 23: Discharged/transferred to a long term care hospital. | 0.41 |
| discharge_disposition_id | 25: Not Mapped | 0.46 |
| discharge_disposition_id | 28: Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital | 1.88 |
| discharge_disposition_id | 5: Discharged/transferred to another type of inpatient care institution | 1.83 |
| discharge_disposition_id | 1: Discharged to home | 0.56 |
| discharge_disposition_id | 15: Discharged/transferred within this institution to Medicare approved swing bed | 1.66 |
| medical_specialty | Orthopedics-Reconstructive | 0.68 |
| diag_2_group | Neoplasms | 1.47 |
| admission_source_id | 3: HMO Referral | 1.44 |

## Model comparison (test set used once)

| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | Brier | Caught in top 10% |
|---|---|---|---|---|---|
| Baseline (always the average rate) | 0.500 ± 0.000 | 0.500 | 0.090 | 0.0817 | 10% |
| Logistic regression | 0.645 ± 0.004 | 0.648 | 0.169 | 0.0793 | 23% |
| Gradient boosting | 0.655 ± 0.006 | 0.654 | 0.186 | 0.0787 | 24% |

Logistic regression C chosen by 5-fold CV: C=0.01: 0.6441, C=0.1: 0.6448, C=1.0: 0.6442 -> C=0.1.

## Fairness check (best model, test set)

| Attribute | Group | n | Observed | Predicted | ROC-AUC |
|---|---|---|---|---|---|
| age | 60-79 | 6,731 | 9.7% | 9.5% | 0.642 |
| age | 40-59 | 3,818 | 7.6% | 7.3% | 0.650 |
| age | 80+ | 2,570 | 10.2% | 10.7% | 0.639 |
| age | <40 | 879 | 6.3% | 6.6% | 0.727 |
| gender | Female | 7,507 | 8.8% | 9.0% | 0.658 |
| gender | Male | 6,491 | 9.1% | 8.8% | 0.650 |
| race | Caucasian | 10,522 | 9.1% | 9.1% | 0.647 |
| race | AfricanAmerican | 2,504 | 9.2% | 8.4% | 0.663 |
| race | Missing | 361 | 6.1% | 8.0% | 0.683 |

Subgroup ROC-AUC ranges from 0.639 to 0.727; predicted and observed rates are listed so miscalibration for any group is visible.

## Cleaning

| Step | Rows left |
|---|---|
| raw encounters | 101,766 |
| drop unknown gender | 101,763 |
| drop expired / hospice discharges | 99,340 |
| keep first encounter per patient | 69,987 |

Dropped columns: weight (about 97% missing); payer_code (about 40% missing and an insurance code, not a clinical signal); encounter_id (identifier); patient_nbr (identifier (used only to keep one encounter per patient)). Constant columns: examide, citoglipton, glimepiride-pioglitazone.

Most-missing raw columns: weight 97%, medical_specialty 49%, payer_code 40%, race 2%, diag_3 1%.

## Limits

- Data from 1999-2008 US hospitals; practice and coding have changed since.
- Administrative fields only: no social factors, follow-up care, or most lab values.
- One encounter per patient keeps the evaluation honest but discards repeat-visit history beyond the counts.
- The calculator is a demonstration, not a clinical tool.
