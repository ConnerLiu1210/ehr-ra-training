# run_readmission.py
# Task: 30-day readmission
# Features: events within first 7 days of each hospital admission
# Tables: LABEVENTS + PRESCRIPTIONS
# Model: PyHealth built-in Transformer
# Metrics: AUC, AUPRC, F1

import os, json, random
from datetime import timedelta
import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer

WINDOW_DAYS = 7
READMIT_DAYS = 30


def get_time(obj, keys):
    # Return the first non-None attribute value from a list of candidate attribute names
    for k in keys:
        v = getattr(obj, k, None)
        if v is not None:
            return v
    return None


def get_event_time(ev):
    # Try common time fields used by different PyHealth event objects
    return get_time(ev, ["timestamp", "charttime", "time", "event_time", "datetime"])


def get_event_code(ev):
    # Try common code fields for labs/medications and return as a string
    c = get_time(ev, ["code", "itemid", "ITEMID", "ndc", "rxnorm", "drug", "drug_name"])
    if c is None:
        return None
    return str(c)


def readmission_7d_30d_fn(patient):
    # Build samples for ONE patient
    # x: codes from LABEVENTS + PRESCRIPTIONS within first 7 days after admission (capped by discharge)
    # y: 1 if next admission occurs within 30 days after discharge, else 0

    samples = []
    obs_window = timedelta(days=WINDOW_DAYS)
    y_window = timedelta(days=READMIT_DAYS)

    # Collect visits and sort by admission time
    visits = []
    for v in patient:
        admit = get_time(v, ["encounter_time", "start_time", "admit_time", "admittime"])
        if admit is not None:
            visits.append((admit, v))
    visits.sort(key=lambda x: x[0])
    visits = [v for _, v in visits]

    for i, visit in enumerate(visits):
        admit = get_time(visit, ["encounter_time", "start_time", "admit_time", "admittime"])
        discharge = get_time(visit, ["discharge_time", "end_time", "dischtime", "discharge_datetime"])
        if admit is None or discharge is None:
            continue

        # Observation end = min(admit + 7 days, discharge)
        obs_end = admit + obs_window
        if discharge < obs_end:
            obs_end = discharge

        # Label from the next admission time
        label = 0
        if i + 1 < len(visits):
            next_visit = visits[i + 1]
            next_admit = get_time(next_visit, ["encounter_time", "start_time", "admit_time", "admittime"])
            if next_admit is not None:
                if discharge < next_admit <= (discharge + y_window):
                    label = 1

        # LABEVENTS codes in [admit, obs_end]
        lab_codes = []
        try:
            lab_evs = visit.get_event_list(table="LABEVENTS")
        except Exception:
            lab_evs = []

        for ev in lab_evs:
            t = get_event_time(ev)
            if t is None:
                continue
            if admit <= t <= obs_end:
                c = get_event_code(ev)
                if c is not None:
                    lab_codes.append(c)

        # PRESCRIPTIONS codes in [admit, obs_end]
        rx_codes = []
        try:
            rx_evs = visit.get_event_list(table="PRESCRIPTIONS")
        except Exception:
            rx_evs = []

        for ev in rx_evs:
            c = get_event_code(ev)
            if c is None:
                continue

            t = get_event_time(ev)

            # If prescription has no timestamp, only include it when stay <= 7 days
            # This avoids leaking prescriptions that could happen after day 7
            if t is None:
                if discharge <= (admit + obs_window):
                    rx_codes.append(c)
                continue

            if admit <= t <= obs_end:
                rx_codes.append(c)

        # Skip admissions with no features
        if (not lab_codes) and (not rx_codes):
            continue

        # PyHealth expects list-of-visits for each feature key
        samples.append(
            {
                "patient_id": patient.patient_id,
                "visit_id": visit.visit_id,
                "labs": [lab_codes] if lab_codes else [[]],
                "rx": [rx_codes] if rx_codes else [[]],
                "label": label,
            }
        )

    return samples


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_root = os.path.expanduser("~/data/mimiciii")

    dataset = MIMIC3Dataset(
        root=data_root,
        tables=["LABEVENTS", "PRESCRIPTIONS"],
    )

    task_dataset = dataset.set_task(readmission_7d_30d_fn)

    train_ds, val_ds, test_ds = split_by_patient(task_dataset, ratios=[0.8, 0.1, 0.1], seed=seed)

    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)

    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else


    model = Transformer(
        dataset=base_ds,
        feature_keys=["labs", "rx"],
        label_key="label",
        mode="binary",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(model=model, device=device, metrics=["roc_auc", "pr_auc", "f1"])

    # Train longer but keep it simple
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=10)

    test_metrics = trainer.evaluate(test_loader)

    out_dir = os.path.join("experiments", "readmission", "results")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "task": "30-day readmission, features=first 7d (capped by discharge) LABEVENTS+PRESCRIPTIONS",
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