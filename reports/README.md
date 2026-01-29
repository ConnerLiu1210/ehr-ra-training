# Dataset Summary: MIMIC-III (via BigQuery)

This repository contains a short dataset summary and basic analysis of the MIMIC-III clinical database using Google BigQuery.

The purpose of this work is to understand the structure of the dataset, cohort statistics, and basic label distributions for structured EHR modeling.

## Dataset

- Dataset: MIMIC-III (v1.4)
- Access: Google BigQuery public dataset
- BigQuery dataset: physionet-data.mimiciii_clinical

MIMIC-III is a de-identified ICU electronic health record (EHR) dataset collected from Beth Israel Deaconess Medical Center (BIDMC). It includes patient demographics, hospital admissions, ICU stays, lab results, medications, diagnoses, and procedures.

## What is included

This file includes:

- Data source, population, and time range
- Available clinical tables
- Cohort statistics (patients, admissions, ICU stays)
- Basic demographics
- Label distributions (mortality, readmission, length of stay, medications)

## Notes

- All analysis is performed using Google BigQuery.
- No raw clinical data is stored in this repository.
- Timestamps in MIMIC-III are de-identified and shifted.
