# Medication Records (Binary) using early admission events

This folder runs a simple baseline to predict whether an admission has any medication record.


## Task

Binary classification:

- 1 = this admission has at least one medication record in PRESCRIPTIONS
- 0 = otherwise


## Observation Window

Use events occurring within the first OBS_DAYS after admission (capped at discharge time).

## Features (Inputs)

Code tokens from tables inside the observation window:

- LABEVENTS item IDs
- DIAGNOSES_ICD codes
- PROCEDURES_ICD codes

Extra categorical tokens:

- Demographics / admission info tokens (gender, ethnicity, insurance, admission type, age bin)
- ICU info tokens (first/last careunit, ICU LOS bins)

## Model

- pyhealth.models.Transformer 

## Metrics

- AUC 
- AUPRC 
- F1 (f1)

## Files

- run_medication.py: training + evaluation script
- results/metrics.json: saved test metrics output

