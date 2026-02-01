# Experiment: In-hospital Mortality Prediction (MIMIC-III)

This folder contains a baseline experiment for predicting in-hospital mortality using MIMIC-III and PyHealth.

## Task
- Level: admission-level
- Objective: predict whether a patient dies during a hospital admission

## Label
- Source: MIMIC-III `ADMISSIONS`
- Label: in-hospital mortality (from the PyHealth task function)
- Positive (1): died in hospital
- Negative (0): survived

## Prediction setting
- Index time: admission time
- Observation window: first 48 hours after admission
- Target: in-hospital mortality outcome

## Features (used by the PyHealth task)
- Diagnoses: `DIAGNOSES_ICD`
- Procedures: `PROCEDURES_ICD`

## Baseline model
- Model: RETAIN (PyHealth built-in)

## Metrics
- AUC (ROC-AUC)
- AUPRC (PR-AUC)
- F1

## Outputs
- experiments/mortality/results/metrics.json