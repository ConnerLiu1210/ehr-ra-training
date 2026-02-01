# run_readmission.py
# Task: 30-day readmission
# Features: events within first 7 days after admission
# Tables: LABEVENTS + PRESCRIPTIONS + DIAGNOSES_ICD + PROCEDURES_ICD
# Model: PyHealth built-in Transformer
# Metrics: AUC, AUPRC, F1

import os
import json
import random
from datetime import timedelta

import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer

WINDOW_DAYS = 7
READMIT_DAYS = 30


def get_first_attr(obj, keys):
    # Return the first non-None attribute from a list of possible names
    for k in keys:
        v = getattr(obj, k, None)
        if v is not None:
            return v
    return None


def get_visit_admit_time(visit):
    return get_first_attr(visit, ["encounter_time", "start_time", "admit_time", "admittime"])


def get_visit_discharge_time(visit):
    return get_first_attr(visit, ["discharge_time", "end_time", "dischtime", "discharge_datetime"])


def get_event_time(ev):
    # Common timestamp fields across different event objects
    return get_first_attr(ev, ["timestamp", "charttime", "time", "event_time", "datetime"])


def get_event_code(ev):
    # Common code fields for labs, meds, diagnoses, procedures
    c = get_first_attr(ev, ["code", "itemid", "ITEMID", "icd_code", "icd9_code", "ndc", "rxnorm", "drug", "drug_name"])
    if c is None:
        return None
    return str(c)


def filter_codes_in_window(events, admit, obs_end, allow_no_time):
    # Collect codes whose event time falls in [admit, obs_end]
    # If time is missing, include only when allow_no_time is True
    codes = []
    for ev in events:
        code = get_event_code(ev)
        if code is None:
            continue

        t = get_event_time(ev)
        if t is None:
            if allow_no_time:
                codes.append(code)
            continue

        if admit <= t <= obs_end:
            codes.append(code)

    return codes


def readmission_7d_30d_fn(patient):
    # Build samples for ONE patient
    # x: codes from multiple tables within first 7 days after admission (capped by discharge)
    # y: 1 if the next admission happens within 30 days after discharge, else 0

    samples = []
    obs_window = timedelta(days=WINDOW_DAYS)
    y_window = timedelta(days=READMIT_DAYS)

    # Sort visits by admission time to define "next admission"
    visits = []
    for v in patient:
        admit = get_visit_admit_time(v)
        if admit is not None:
            visits.append((admit, v))
    visits.sort(key=lambda x: x[0])
    visits = [v for _, v in visits]

    for i, visit in enumerate(visits):
        admit = get_visit_admit_time(visit)
        discharge = get_visit_discharge_time(visit)
        if admit is None or discharge is None:
            continue

        # Observation end = min(admit + 7 days, discharge)
        obs_end = admit + obs_window
        if discharge < obs_end:
            obs_end = discharge

        # If the whole stay is <= 7 days, then "no timestamp" items are safe to include
        stay_leq_7d = discharge <= (admit + obs_window)

        # Label: readmitted within 30 days after discharge
        label = 0
        if i + 1 < len(visits):
            next_admit = get_visit_admit_time(visits[i + 1])
            if next_admit is not None:
                if discharge < next_admit <= (discharge + y_window):
                    label = 1

        # LABEVENTS
        try:
            lab_evs = visit.get_event_list(table="LABEVENTS")
        except Exception:
            lab_evs = []
        lab_codes = filter_codes_in_window(lab_evs, admit, obs_end, allow_no_time=False)

        # PRESCRIPTIONS
        try:
            rx_evs = visit.get_event_list(table="PRESCRIPTIONS")
        except Exception:
            rx_evs = []
        rx_codes = filter_codes_in_window(rx_evs, admit, obs_end, allow_no_time=stay_leq_7d)

        # DIAGNOSES_ICD
        try:
            dx_evs = visit.get_event_list(table="DIAGNOSES_ICD")
        except Exception:
            dx_evs = []
        dx_codes = filter_codes_in_window(dx_evs, admit, obs_end, allow_no_time=stay_leq_7d)

        # PROCEDURES_ICD
        try:
            px_evs = visit.get_event_list(table="PROCEDURES_ICD")
        except Exception:
            px_evs = []
        px_codes = filter_codes_in_window(px_evs, admit, obs_end, allow_no_time=stay_leq_7d)

        # Skip if no usable features
        if (not lab_codes) and (not rx_codes) and (not dx_codes) and (not px_codes):
            continue

        # PyHealth expects list-of-visits for each feature key
        samples.append(
            {
                "patient_id": patient.patient_id,
                "visit_id": visit.visit_id,
                "labs": [lab_codes] if lab_codes else [[]],
                "rx": [rx_codes] if rx_codes else [[]],
                "dx": [dx_codes] if dx_codes else [[]],
                "px": [px_codes] if px_codes else [[]],
                "label": label,
            }
        )

    return samples


def main():
    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_root = os.path.expanduser("~/data/mimiciii")

    # Load tables (adding DIAGNOSES_ICD and PROCEDURES_ICD)
    dataset = MIMIC3Dataset(
        root=data_root,
        tables=["LABEVENTS", "PRESCRIPTIONS", "DIAGNOSES_ICD", "PROCEDURES_ICD"],
    )

    # Convert patient objects to (feature, label) samples
    task_dataset = dataset.set_task(readmission_7d_30d_fn)

    # Patient-level split to avoid leakage across admissions of the same patient
    train_ds, val_ds, test_ds = split_by_patient(task_dataset, ratios=[0.8, 0.1, 0.1], seed=seed)

    # Dataloaders
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)

    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds

    # Model
    model = Transformer(
        dataset=base_ds,
        feature_keys=["labs", "rx", "dx", "px"],
        label_key="label",
        mode="binary",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Trainer
    trainer = Trainer(model=model, device=device, metrics=["roc_auc", "pr_auc", "f1"])

    # Train
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=10)

    # Evaluate
    test_metrics = trainer.evaluate(test_loader)

    # Save metrics
    out_dir = os.path.join("experiments", "readmission", "results")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "task": "30-day readmission, features=first 7d LABEVENTS+PRESCRIPTIONS+DIAGNOSES_ICD+PROCEDURES_ICD",
        "model": "Transformer",
        "window_days": WINDOW_DAYS,
        "readmit_days": READMIT_DAYS,
        "metrics": {
            "auc": float(test_metrics.get("roc_auc", np.nan)),
            "auprc": float(test_metrics.get("pr_auc", np.nan)),
            "f1": float(test_metrics.get("f1", np.nan)),
        },
    }

    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Test metrics:", results["metrics"])
    print("Saved:", out_path)


if __name__ == "__main__":
    main()