# Dataset Summary: MIMIC-III (via BigQuery)

## 1) Dataset chosen
I use the MIMIC-III dataset (BigQuery public dataset: `physionet-data.mimiciii_*`).


## 2) Data source, population, and time range

<img width="840" height="340" alt="截屏2026-01-29 14 03 10" src="https://github.com/user-attachments/assets/036f05f3-0cd5-45fe-96a4-d06f4bb49a15" />

- Data source: MIMIC-III is a de-identified ICU electronic health record (EHR) dataset collected from Beth Israel Deaconess Medical Center (BIDMC).
- Population: ICU patients, including hospital admissions, ICU stays, lab tests, medications, and procedures.
- Time range (from the `admissions` table):
  - Earliest admittime: 2100-06-07  
  - Latest admittime: 2210-08-17  


## 3) Available tables

<img width="834" height="745" alt="截屏2026-01-29 14 04 03" src="https://github.com/user-attachments/assets/ea637083-0889-401c-9943-fe57246f2ebf" />
<img width="831" height="247" alt="截屏2026-01-29 14 04 15" src="https://github.com/user-attachments/assets/c5893d65-f068-4533-8af7-e61b3412c5b0" />

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


## 4) Cohort statistics

<img width="835" height="339" alt="截屏2026-01-29 13 57 09" src="https://github.com/user-attachments/assets/6dbae1b0-b34e-4b9e-b2d1-e742961645f6" />

<img width="839" height="334" alt="截屏2026-01-29 13 58 40" src="https://github.com/user-attachments/assets/cd48463a-183d-41fa-aa4c-983e75eff2ab" />

<img width="842" height="339" alt="截屏2026-01-29 14 00 48" src="https://github.com/user-attachments/assets/97907726-33c5-4abb-b1ab-02ec04e9f8ac" />

From BigQuery:
- Number of patients: 46,520
- Number of hospital admissions: 58,976
- Number of ICU stays: 61,532


## 5) Basic demographics & labels

<img width="840" height="340" alt="截屏2026-01-29 14 03 10" src="https://github.com/user-attachments/assets/b8f9b39f-edfc-440d-8e74-cafa030bcf47" />

Gender distribution (from `patients`):
- Female: 20,399 (43.85%)
- Male: 26,121 (56.15%)

  
## 6） Label1 distribution (mortality)

<img width="835" height="369" alt="截屏2026-01-29 15 37 21" src="https://github.com/user-attachments/assets/599d7018-09db-49b1-9572-14965c4cfcf8" />

Based on `hospital_expire_flag` in the `admissions` table:

- Death (label = 1): 5,854 admissions (9.93%)
- Survived (label = 0): 53,122 admissions (90.07%)

The dataset is imbalanced, with fewer positive (death) cases.

## 7） Label2 distribution (readmission)

<img width="836" height="516" alt="截屏2026-01-29 15 55 18" src="https://github.com/user-attachments/assets/fea3466f-ec06-4173-a8d2-9e831f1538bb" />

Based on multiple admissions per patient (using admissions table):

- Not readmitted (label2 = 0): 38,983 admissions  
- Readmitted (label2 = 1): 19,993 admissions  

Interpretation:  
Admissions from patients with more than one hospital admission are labeled as readmitted. 

## 8) Label3 distribution (length of stay, LOS)

<img width="838" height="489" alt="截屏2026-01-29 16 28 31" src="https://github.com/user-attachments/assets/5b9dc30f-9edc-491d-a2a6-ea164b4333bc" />

Based on hospital length of stay from the admissions table:

- Mean LOS: 10.1 days  
- Q1: 4 days  
- Median: 6 days  
- Q3: 12 days  

Interpretation:  
Most admissions are shorter than one week, but some long stays increase the average LOS.

## 9) Label4 distribution (medication records)

<img width="836" height="489" alt="截屏2026-01-29 16 43 15" src="https://github.com/user-attachments/assets/403e0e17-f11a-4098-8bef-f488fa9239b5" />

<img width="836" height="490" alt="截屏2026-01-29 16 45 08" src="https://github.com/user-attachments/assets/12fe1cfe-16a5-4128-ae81-24c7beaca307" />

Based on medication records from the prescriptions table (BigQuery):

- Admissions with at least one medication record: 50,216 admissions
- Average number of medication orders per admission: 82.8

Interpretation:  
Most hospital admissions include medication records, and patients typically receive a large number of medication orders during a single hospital stay, reflecting the complexity of ICU and inpatient care.
