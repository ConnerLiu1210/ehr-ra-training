# Task: In-hospital Mortality Prediction (MIMIC-III)

## 1. Task Definition and Label
- Level: admission-level
- Objective: predict whether a patient dies during a hospital admission
- Label source table: ADMISSIONS
- Label name: hospital_expire_flag
- Positive class (1): patient died in hospital
- Negative class (0): patient survived
- Class imbalance:  10% positive, 90% negative

## 2. Features and Time Window
- Index time: hospital admission time
- Observation window: first 24 hours after admission
- Prediction target: in-hospital mortality
- Data sources:
  - Diagnoses (DIAGNOSES_ICD)
  - Laboratory events (LABEVENTS)
  - Medications (PRESCRIPTIONS)

## 3. Baseline Model Results
- Model: PyHealth baseline model
- AUC, AUPRC, F1

