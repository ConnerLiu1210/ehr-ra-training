# Readmission

This folder runs a simple 30-day readmission baseline using events from the first 7 days of each hospital admission.

## Task

Predict 30-day readmission after discharge:

- 1 = has a next admission within 30 days after discharge
- 0 = otherwise

## Features

Events occurring within the first 7 days after admission (capped at discharge time):

- LABEVENTS item IDs
- PRESCRIPTIONS codes
- DIAGNOSES_ICD codes
- PROCEDURES_ICD codes

## Extra simple categorical tokens (when available):
- Demographics/admission tokens (gender, ethnicity, insurance, admission_type, age_bin)
- ICU tokens (first_careunit, last_careunit, ICU length-of-stay bins)


## Model

- pyhealth.models.Transformer

## Metrics

- auc
- auprc
- f1
- Note: F1 can be very low under class imbalance if using the default threshold (often 0.5).


## Files

- run_readmission.py: training + evaluation script
- results/metrics.json: saved test metrics output
