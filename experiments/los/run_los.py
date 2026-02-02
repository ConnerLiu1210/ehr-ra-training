# run_los.py
# Task: Length of Stay (LOS) prediction as a binary classification
# Observation window: first N days after admission (capped by discharge)
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

# Settings you can tune
OBS_WINDOW_DAYS = 2
LOS_THRESHOLD_DAYS = 7       # long stay threshold: LOS >= 7 days -> label 1
MAX_CODES_PER_TABLE = 200    # cap per table per visit to avoid huge sequences

# Empty tokens to avoid empty feature lists (prevents validation issues)
EMPTY_LAB = "__EMPTY_LAB__"
EMPTY_RX = "__EMPTY_RX__"
EMPTY_DX = "__EMPTY_DX__"
EMPTY_PX = "__EMPTY_PX__"


def get_first_attr(obj, keys):
    # Return the first non-None attribute from a list of possible names
    for k in keys:
        v = getattr(obj, k, None)
        if v is not None:
            return v
    return None


def dedup_and_cap(codes, max_len):
    # Remove duplicates while keeping order, then cap length
    seen = set()
    out = []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= max_len:
            break
    return out


def wrap_feature(codes, empty_token):
    # PyHealth expects list-of-visits: [codes_for_this_visit]
    # Never return [[]] because it can trigger validation errors
    if codes is None or len(codes) == 0:
        return [[empty_token]]
    return [codes]


def get_visit_admit_time(visit):
    return get_first_attr(visit, ["encounter_time", "start_time", "admit_time", "admittime"])


def get_visit_discharge_time(visit):
    return get_first_attr(visit, ["discharge_time", "end_time", "dischtime", "discharge_datetime"])


def get_event_time(ev):
    # Common timestamp fields across different event objects
    return get_first_attr(ev, ["timestamp", "charttime", "time", "event_time", "datetime"])


def get_event_code(ev):
    # Common code fields for labs, meds, diagnoses, procedures
    c = get_first_attr(
        ev,
        ["code", "itemid", "ITEMID", "icd_code", "icd9_code", "ndc", "rxnorm", "drug", "drug_name"],
    )
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

    return dedup_and_cap(codes, MAX_CODES_PER_TABLE)


def los_early_fn(patient):
    # Build samples for ONE patient
    # x: codes within first OBS_WINDOW_DAYS after admission
    # y: 1 if LOS >= LOS_THRESHOLD_DAYS else 0

    samples = []
    obs_window = timedelta(days=OBS_WINDOW_DAYS)
    los_threshold = timedelta(days=LOS_THRESHOLD_DAYS)

    # Sort visits by admission time
    visits = []
    for v in patient:
        a = get_visit_admit_time(v)
        if a is not None:
            visits.append((a, v))
    visits.sort(key=lambda x: x[0])
    visits = [v for _, v in visits]

    for visit in visits:
        admit = get_visit_admit_time(visit)
        discharge = get_visit_discharge_time(visit)
        if admit is None or discharge is None:
            continue

        # LOS and label
        los = discharge - admit
        label = 1 if los >= los_threshold else 0

        # Observation end = min(admit + obs_window, discharge)
        obs_end = admit + obs_window
        if discharge < obs_end:
            obs_end = discharge

        stay_within_obs = discharge <= (admit + obs_window)

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
        rx_codes = filter_codes_in_window(rx_evs, admit, obs_end, allow_no_time=stay_within_obs)

        # DIAGNOSES_ICD
        try:
            dx_evs = visit.get_event_list(table="DIAGNOSES_ICD")
        except Exception:
            dx_evs = []
        dx_codes = filter_codes_in_window(dx_evs, admit, obs_end, allow_no_time=stay_within_obs)

        # PROCEDURES_ICD
        try:
            px_evs = visit.get_event_list(table="PROCEDURES_ICD")
        except Exception:
            px_evs = []
        px_codes = filter_codes_in_window(px_evs, admit, obs_end, allow_no_time=stay_within_obs)

        # Skip admissions with absolutely no usable features
        if len(lab_codes) == 0 and len(rx_codes) == 0 and len(dx_codes) == 0 and len(px_codes) == 0:
            continue

        samples.append(
            {
                "patient_id": patient.patient_id,
                "visit_id": visit.visit_id,
                "labs": wrap_feature(lab_codes, EMPTY_LAB),
                "rx": wrap_feature(rx_codes, EMPTY_RX),
                "dx": wrap_feature(dx_codes, EMPTY_DX),
                "px": wrap_feature(px_codes, EMPTY_PX),
                "label": int(label),
            }
        )

    return samples


def get_labels_from_dataset(ds):
    # split_by_patient may return a Subset-like object
    if hasattr(ds, "samples"):
        return [s["label"] for s in ds.samples]

    # Try Subset pattern: ds.dataset.samples + ds.indices
    base = getattr(ds, "dataset", None)
    idxs = getattr(ds, "indices", None)
    if base is not None and idxs is not None and hasattr(base, "samples"):
        return [base.samples[i]["label"] for i in idxs]

    return []


def main():
    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_root = os.path.expanduser("~/data/mimiciii")

    dataset = MIMIC3Dataset(
        root=data_root,
        tables=["LABEVENTS", "PRESCRIPTIONS", "DIAGNOSES_ICD", "PROCEDURES_ICD"],
    )

    task_dataset = dataset.set_task(los_early_fn)

    # Patient-level split
    train_ds, val_ds, test_ds = split_by_patient(task_dataset, ratios=[0.8, 0.1, 0.1], seed=seed)

    # Label prevalence check
    y_train = get_labels_from_dataset(train_ds)
    pos_rate = float(np.mean(y_train)) if len(y_train) > 0 else float("nan")
    print("Train label=1 rate:", pos_rate, "Train count:", len(y_train))

    # Dataloaders
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)

    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds

    model = Transformer(
        dataset=base_ds,
        feature_keys=["labs", "rx", "dx", "px"],
        label_key="label",
        mode="binary",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model=model, device=device, metrics=["roc_auc", "pr_auc", "f1"])

    # Train
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=10)

    # Evaluate
    test_metrics = trainer.evaluate(test_loader)

    # Save results
    out_dir = os.path.join("experiments", "los", "results")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "task": "LOS binary prediction",
        "label": f"LOS >= {LOS_THRESHOLD_DAYS} days",
        "observation_window_days": OBS_WINDOW_DAYS,
        "tables": ["LABEVENTS", "PRESCRIPTIONS", "DIAGNOSES_ICD", "PROCEDURES_ICD"],
        "model": "Transformer",
        "max_codes_per_table": MAX_CODES_PER_TABLE,
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