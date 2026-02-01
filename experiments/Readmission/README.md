##Readmission

This folder runs a simple 30-day readmission baseline using events from the first 7 days of each hospital admission.

##Task

Predict 30-day readmission after discharge:

- 1 = has a next admission within 30 days after discharge
- 0 = otherwise

##Features

Events occurring within the first 7 days after admission (capped at discharge time):

- LABEVENTS item IDs
- PRESCRIPTIONS codes

##Model

- pyhealth.models.Transformer

##Metrics

- auc
- auprc
- f1

##Files

- run_readmission.py: training + evaluation script
- results/metrics.json: saved test metrics output
