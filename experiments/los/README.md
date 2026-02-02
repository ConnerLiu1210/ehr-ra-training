# LOS(length of stay)

This folder runs a simple LOS baseline as a binary classification task using early admission events.

## Task

Predict whether the hospital length of stay is long:

- 1 = LOS >= LOS_THRESHOLD_DAYS
- 0 = LOS < LOS_THRESHOLD_DAYS

LOS is computed as (discharge_time - admit_time) in days.

## Observation Window

Use events in the first OBS_DAYS after admission (capped at discharge time).

## Features

Code tokens from multiple tables inside the observation window:

- LABEVENTS item IDs
- PRESCRIPTIONS codes
- DIAGNOSES_ICD codes
- PROCEDURES_ICD codes
- CHARTEVENTS item IDs

Extra categorical tokens (when available):

- Demographics / admission info tokens (gender, ethnicity, insurance, admission type, age bin)
- ICU info tokens (first/last careunit, ICU LOS bins)

To keep sequences stable:

- Deduplicate codes per table per visit
- Cap codes per table per visit with MAX_CODES_PER_TABLE
- If a feature list is empty, insert a placeholder token (prevents PyHealth validation errors)

## Model

- pyhealth.models.Transformer

## Metrics

- roc_auc
- pr_auc
- f1

## Files

- run_los.py: training + evaluation script
- results/metrics.json: saved test metrics output
