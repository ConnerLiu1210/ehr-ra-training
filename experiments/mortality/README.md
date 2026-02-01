# In-hospital Mortality Baseline (MIMIC-III)

This folder contains a simple baseline experiment for predicting in-hospital mortality using MIMIC-III and PyHealth.

## Task
- Dataset: MIMIC-III 
- Level: admission-level
- Label: in-hospital mortality (from `hospital_expire_flag` via PyHealth task)
- Observation window: first 48 hours after admission (`prediction_window = 48 * 60` minutes)

## Model
- Model: Transformer (`pyhealth.models.Transformer`)

## Metrics
- ROC-AUC
- PR-AUC (AUPRC)
- F1

## Files
- `run_mortality.py`: trains and evaluates the Transformer baseline
- `experiments/mortality/results/metrics.json`: saved test metrics output

After running, check:
- `experiments/mortality/results/metrics.json`