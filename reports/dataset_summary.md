# Dataset Summary: MIMIC-III (via BigQuery)

## 1) Dataset chosen
I use the MIMIC-III dataset (BigQuery public dataset: 

<img width="840" height="340" alt="截屏2026-01-29 14 03 10" src="https://github.com/user-attachments/assets/0e7a95dd-81e4-443f-8738-e2a8b68a670b" />

## 2) Data source, population, and time range
- Data source: MIMIC-III is a de-identified ICU electronic health record (EHR) dataset collected from Beth Israel Deaconess Medical Center (BIDMC).
- Population: ICU patients, including hospital admissions, ICU stays, lab tests, medications, and procedures.
- Time range (from the `admissions` table):
  - Earliest admittime: 2100-06-07  
  - Latest admittime: 2210-08-17  

  Note: All timestamps in MIMIC are de-identified and shifted. The absolute years are not real, but relative timing and durations are still meaningful.

<img width="834" height="745" alt="截屏2026-01-29 14 04 03" src="https://github.com/user-attachments/assets/6c48474c-e8b5-48da-a50e-702b0733f1c2" />
<img width="831" height="247" alt="截屏2026-01-29 14 04 15" src="https://github.com/user-attachments/assets/3731e5bc-21ab-431e-8195-0d87a52c892e" />

## 3) Available tables
Main clinical tables (`mimiciii_clinical`) include:
- patients: patient demographics (e.g., gender)
- admissions: hospital admissions and timing
- icustays: ICU stay information
- chartevents: bedside vital signs and charted measurements
- labevents: laboratory results
- prescriptions: structured medication orders
- diagnoses_icd / procedures_icd: diagnosis and procedure codes
- Other tables such as services, transfers, drgcodes, caregivers, etc.
- Dictionary tables such as d_items, d_labitems, d_icd_diagnoses, d_icd_procedures, and d_cpt

Clinical notes are stored in separate datasets:
- `mimiciii_notes` and `mimiciii_notes_derived`, which include tables such as noteevents.

<img width="835" height="339" alt="截屏2026-01-29 13 57 09" src="https://github.com/user-attachments/assets/23f4b210-2f22-4d59-a55c-86e3d93e83d5" />
<img width="839" height="334" alt="截屏2026-01-29 13 58 40" src="https://github.com/user-attachments/assets/999de3c2-c03a-4ab0-af2a-05acaa0162c1" />
<img width="842" height="339" alt="截屏2026-01-29 14 00 48" src="https://github.com/user-attachments/assets/8c6d5bcb-6d9a-4ab7-9ae6-c7daa3bb8ba4" />

## 4) Cohort statistics
From BigQuery:
- Number of patients: 46,520
- Number of hospital admissions: 58,976
- Number of ICU stays: 61,532

In this project:
- Hospital admissions (hadm_id) are treated as visits.
- ICU stays (icustay_id) are treated as encounters for ICU-level tasks.

<img width="842" height="363" alt="截屏2026-01-29 14 01 17" src="https://github.com/user-attachments/assets/e3acb1ba-fe22-43c0-a4ba-c314e1a69416" />

## 5) Basic demographics & labels
Gender distribution (from `patients`):
- Female: 20,399 (43.85%)
- Male: 26,121 (56.15%)

