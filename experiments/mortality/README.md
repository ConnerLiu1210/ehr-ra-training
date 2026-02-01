# Mortality

This folder runs a simple in-hospital mortality baseline using LABEVENTS + CHARTEVENTS from the first 48 hours of each hospital admission.

## Task

Predict in-hospital mortality:

- 1 = died in hospital
- 0 = survived / other

## Features

- LABEVENTS item IDs occurring within the first 48 hours after admission
- CHARTEVENTS item IDs (chart / vitals-related events) occurring within the first 48 hours after admission


## Model

- pyhealth.models.Transformer

## Metrics

- auc
- auprc
- f1

## Files

- run_mortality.py: training + evaluation script
- results/metrics.json: saved test metrics output