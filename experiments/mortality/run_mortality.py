# run_mortality.py
# Task: In-hospital mortality
# Features: LABEVENTS + CHARTEVENTS within first 48 hours of each hospital admission
# Model: PyHealth built-in Transformer
# Metrics: AUC, AUPRC, F1

import os, json, random
from datetime import timedelta
import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer

# Prediction window length
WINDOW_HOURS = 48


def mortality_48h_labs_vitals_fn(patient):
    # Build a list of training samples for ONE patient
    samples = []
    window = timedelta(hours=WINDOW_HOURS)

    # patient is iterable over visits (hospital admissions)
    for i in range(len(patient)):
        visit = patient[i]

        # Admission start time
        admit = getattr(visit, "encounter_time", None) or getattr(visit, "start_time", None)
        if admit is None:
            continue

        # End time of the observation window
        end = admit + window

        # Label: 1 if died in hospital, 0 otherwise
        ds = getattr(visit, "discharge_status", None)
        label = int(ds) if ds in [0, 1] else 0

        # LABEVENTS
        try:
            lab_evs = visit.get_event_list(table="LABEVENTS")
        except Exception:
            lab_evs = []

        lab_codes = []
        for ev in lab_evs:
            t = getattr(ev, "timestamp", None) or getattr(ev, "charttime", None) or getattr(ev, "time", None)
            if t is None:
                continue
            if admit <= t <= end:
                c = getattr(ev, "code", None) or getattr(ev, "itemid", None) or getattr(ev, "ITEMID", None)
                if c is not None:
                    lab_codes.append(str(c))

        # CHARTEVENTS
        try:
            chart_evs = visit.get_event_list(table="CHARTEVENTS")
        except Exception:
            chart_evs = []

        vital_codes = []
        for ev in chart_evs:
            t = getattr(ev, "timestamp", None) or getattr(ev, "charttime", None) or getattr(ev, "time", None)
            if t is None:
                continue
            if admit <= t <= end:
                c = getattr(ev, "code", None) or getattr(ev, "itemid", None) or getattr(ev, "ITEMID", None)
                if c is not None:
                    vital_codes.append(str(c))

        # Skip visits with no events in the window
        if (not lab_codes) and (not vital_codes):
            continue

        samples.append(
            {
                "patient_id": patient.patient_id,
                "visit_id": visit.visit_id,
                "labs": [lab_codes] if lab_codes else [[]],
                "vitals": [vital_codes] if vital_codes else [[]],
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

    # Load MIMIC-III tables
    dataset = MIMIC3Dataset(
        root=data_root,
        tables=["LABEVENTS", "CHARTEVENTS"],
    )

    # Apply the task function to convert raw visits into (x, y) samples
    task_dataset = dataset.set_task(mortality_48h_labs_vitals_fn)

    # Split by patient
    train_ds, val_ds, test_ds = split_by_patient(task_dataset, ratios=[0.8, 0.1, 0.1], seed=seed)

    # Wrap datasets into dataloaders
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)
    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds

    # Initialize Transformer using dataset metadata
    model = Transformer(
        dataset=base_ds,
        feature_keys=["labs", "vitals"],
        label_key="label",
        mode="binary",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Trainer handles training loop + metric computation
    trainer = Trainer(model=model, device=device, metrics=["roc_auc", "pr_auc", "f1"])

    # Train
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=5)

    # Evaluate
    test_metrics = trainer.evaluate(test_loader)

    # Save metrics
    out_dir = os.path.join("experiments", "mortality", "results")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "task": "in-hospital mortality, features=first 48h LABEVENTS + CHARTEVENTS",
        "model": "Transformer",
        "window_hours": WINDOW_HOURS,
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