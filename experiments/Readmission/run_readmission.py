# run_readmission.py
# Task: 30-day readmission
# Features: events within first 7 days of each hospital admission
# Model: PyHealth built-in Transformer
# Metrics: AUC, AUPRC, F1

import os, json, random
from datetime import timedelta
import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer

# Observation window: first 7 days after admission
WINDOW_DAYS = 7

# Outcome window: readmitted within 30 days after discharge
READMIT_DAYS = 30


def _get_time(obj, keys):
    for k in keys:
        v = getattr(obj, k, None)
        if v is not None:
            return v
    return None


def _get_event_time(ev):
    return _get_time(ev, ["timestamp", "charttime", "time", "event_time", "datetime"])


def _get_event_code(ev):
    c = _get_time(ev, ["code", "itemid", "ITEMID", "drug", "drug_name", "ndc", "rxnorm", "medication"])
    return None if c is None else str(c)


def readmission_7d_30d_fn(patient):
    samples = []
    obs_window = timedelta(days=WINDOW_DAYS)
    y_window = timedelta(days=READMIT_DAYS)

    # iterate each admission
    for i in range(len(patient)):
        visit = patient[i]

        # admission start time
        admit = _get_time(visit, ["encounter_time", "start_time", "admit_time", "admittime"])
        if admit is None:
            continue

        # discharge / end time
        discharge = _get_time(visit, ["discharge_time", "end_time", "dischtime", "discharge_datetime"])
        if discharge is None:
            # without discharge, cannot define the 30-day post-discharge label
            continue

        # observation end: min(admit+7d, discharge)
        obs_end = admit + obs_window
        if discharge < obs_end:
            obs_end = discharge

        # label: whether next admission is within 30 days after discharge
        label = 0
        if i + 1 < len(patient):
            next_visit = patient[i + 1]
            next_admit = _get_time(next_visit, ["encounter_time", "start_time", "admit_time", "admittime"])
            if next_admit is not None:
                # readmission should be after discharge, and within discharge + 30 days
                if discharge < next_admit <= (discharge + y_window):
                    label = 1

        # collect LABEVENTS within observation window
        lab_codes = []
        try:
            lab_evs = visit.get_event_list(table="LABEVENTS")
        except Exception:
            lab_evs = []

        for ev in lab_evs:
            t = _get_event_time(ev)
            if t is None:
                continue
            if admit <= t <= obs_end:
                c = _get_event_code(ev)
                if c is not None:
                    lab_codes.append(c)

        # collect PRESCRIPTIONS within observation window
        rx_codes = []
        try:
            rx_evs = visit.get_event_list(table="PRESCRIPTIONS")
        except Exception:
            rx_evs = []

        for ev in rx_evs:
            t = _get_event_time(ev)
            if t is None:
                # PRESCRIPTIONS sometimes has no event time in some pipelines; skip if missing
                continue
            if admit <= t <= obs_end:
                c = _get_event_code(ev)
                if c is not None:
                    rx_codes.append(c)

        # skip admissions with no features at all
        if (not lab_codes) and (not rx_codes):
            continue


        # wrap the codes as a single-visit sequence
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
    # reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_root = os.path.expanduser("~/data/mimiciii")

    # load MIMIC-III tables
    dataset = MIMIC3Dataset(
        root=data_root,
        tables=["LABEVENTS", "PRESCRIPTIONS"],
    )

    # apply task function -> (x, y) samples
    task_dataset = dataset.set_task(readmission_7d_30d_fn)

    # patient-level split
    train_ds, val_ds, test_ds = split_by_patient(task_dataset, ratios=[0.8, 0.1, 0.1], seed=seed)

    # dataloaders
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)

    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds

    # model
    model = Transformer(
        dataset=base_ds,
        feature_keys=["labs", "rx"],
        label_key="label",
        mode="binary",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # trainer
    trainer = Trainer(model=model, device=device, metrics=["roc_auc", "pr_auc", "f1"])

    # train
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=5)

    # eval
    test_metrics = trainer.evaluate(test_loader)

    # save metrics
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