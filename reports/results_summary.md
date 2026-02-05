# Analyze results

This section summarizes and explains the baseline results from PyHealth built-in model `Transformer` on four tasks: LOS, medication record, mortality, and readmission.

---

## 1. Performance comparison across tasks

All four experiments used PyHealth Transformer. The metrics are reported as AUC, AUPRC, and F1.

### LOS (binary, LOS ≥ 7 days)
- AUC: 0.7880  
- AUPRC: 0.7438  
- F1: 0.7028  

### Medication record (binary, has PRESCRIPTIONS record)
- AUC: 0.9411  
- AUPRC: 0.9882  
- F1: 0.9486  

### In-hospital mortality (binary)
- AUC: 0.8105  
- AUPRC: 0.3482  
- F1: 0.2351  

### 30-day readmission (binary)
- AUC: 0.6147  
- AUPRC: 0.0894  
- F1: 0.0499  

### Overall ranking (easiest -> hardest)
Medication record > LOS > Mortality > Readmission

---

## 2. Why some tasks work better than others

### (1) Medication record is extremely high
This task is basically predicting whether an admission has any record in the `PRESCRIPTIONS` table. In MIMIC-III, most admissions have medication records, so the label is strongly connected to “normal hospital workflow + having enough events recorded”.

Also, early signals like many labs/diagnoses/procedures often imply that medications were ordered, so it becomes an easier classification problem.  
So the high score is not necessarily “clinical prediction is perfect”, it’s more like the dataset/logging pattern makes it easier.

### (2) LOS is decent
LOS relates to severity and complexity. Using early events (first 2 days) plus multiple tables (labs, diagnoses, procedures, prescriptions) gives the model a lot of useful signals about patient condition and treatment intensity.

However, LOS is also affected by non-medical factors like discharge planning, transfer, social factors, etc. So it’s harder than “record exists” tasks.

### (3) Mortality: AUC is okay but AUPRC/F1 are low
Mortality is usually a rare outcome (class imbalance). With rare positives:
- AUPRC is naturally harder to be high.
- F1 depends on threshold and how many positives the model catches.

Using only early labs can still rank risk (so AUC is decent), but it’s harder to correctly identify positive cases with high precision/recall, which causes low AUPRC and F1.

### (4) Readmission is the hardest
30-day readmission depends on many things outside the hospital stay:
follow-up care, social support, insurance, medication adherence, outpatient resources, etc.

Even with 7-day features plus demo/ICU tokens, the model cannot see the full picture, so performance is low. This is expected for readmission tasks.

---

## 3. Most predictive features (what likely matters)


### LOS (first 2 days, multiple tables)
Strong signals usually come from severity and treatment intensity:
- LABEVENTS: abnormal labs (kidney function, infection markers, acid-base, etc.)
- DIAGNOSES_ICD: serious diagnoses / complications
- PROCEDURES_ICD: major procedures (often increases length of stay)
- PRESCRIPTIONS: stronger treatments often correlate with longer stay

### Medication record (label = PRESCRIPTIONS exists)
This is not predicting a specific drug.
It’s mostly predicting whether the admission generates medication records at all.
So signals like “more active care / more events / more diagnoses/procedures” are enough to predict it.

### Mortality (first 48 hours labs)
Mortality risk can be related to extreme abnormal labs (organ failure, shock, severe infection).  
The model can rank risk (AUC), but catching rare positive cases is still difficult.

### Readmission (first 7 days + demo/ICU tokens)
Possible useful signals:
- chronic disease burden (diagnosis patterns)
- ICU stay / severity indicators
- discharge condition not captured well in structured data
But overall readmission has high noise and missing outside-hospital factors, so the model struggles.

---

## 4. What worked / what didn’t
### What worked
- Multi-table features helped LOS.
- Early labs gave reasonable ranking power for mortality (AUC).
- Medication record task is very easy to get high scores.

### What didn’t work (or limited)
- Readmission performance is limited by missing post-discharge information.
- Mortality AUPRC/F1 are limited by class imbalance and thresholding, even if AUC looks okay.

---

## Short conclusion
Medication record prediction got the best metrics because the label is closely tied to dataset 
recording behavior. LOS is moderate because it reflects severity and care complexity but still has 
non-medical factors. Mortality is harder due to rare positives (low AUPRC/F1). Readmission is the 
hardest because many causes are outside the hospital stay and not captured in the available tables.
