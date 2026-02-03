# Medication record prediction (binary)
# Label: 1 if this admission has any PRESCRIPTIONS record, else 0
# Features: early admission events within first OBS_DAYS
# Tables: LABEVENTS + DIAGNOSES_ICD + PROCEDURES_ICD
# Model: PyHealth Transformer
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

OBS_DAYS = 2
MAX_CODES_PER_TABLE = 200

EMPTY_LAB = "__EMPTY_LAB__"
EMPTY_DX = "__EMPTY_DX__"
EMPTY_PX = "__EMPTY_PX__"
EMPTY_DEMO = "__EMPTY_DEMO__"


def first_attr(obj, keys):
    # Return the first non-None attribute among keys
    for k in keys:
        v = getattr(obj, k, None)
        if v is not None:
            return v
    return None


def dedup_cap(seq, max_len):
    # Deduplicate while keeping order, then cap length
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= max_len:
            break
    return out


def wrap_codes(codes, empty_token):
    if not codes:
        return [[empty_token]]
    return [codes]


def visit_admit(visit):
    return first_attr(visit, ["encounter_time", "start_time", "admit_time", "admittime"])


def visit_discharge(visit):
    return first_attr(visit, ["discharge_time", "end_time", "dischtime", "discharge_datetime"])


def event_time(ev):
    return first_attr(ev, ["timestamp", "charttime", "time", "event_time", "datetime"])


def event_code(ev):
    # Common fields across different event objects
    c = first_attr(
        ev,
        ["code", "itemid", "ITEMID", "icd_code", "icd9_code", "ndc", "rxnorm", "drug", "drug_name"],
    )
    if c is None:
        return None
    return str(c)


def codes_in_window(visit, table, admit_t, end_t, allow_no_time):
    # Collect codes in [admit_t, end_t]
    try:
        evs = visit.get_event_list(table=table)
    except Exception:
        evs = []

    codes = []
    for ev in evs:
        c = event_code(ev)
        if c is None:
            continue

        t = event_time(ev)
        if t is None:
            if allow_no_time:
                codes.append(c)
            continue

        if admit_t <= t <= end_t:
            codes.append(c)

    return dedup_cap(codes, MAX_CODES_PER_TABLE)


def demo_tokens(patient, visit, admit_t):
    # Simple categorical tokens
    toks = []

    gender = first_attr(patient, ["gender", "sex"])
    if gender is not None:
        toks.append("gender_" + str(gender))

    ethnicity = first_attr(visit, ["ethnicity"])
    if ethnicity is not None:
        toks.append("eth_" + str(ethnicity))

    insurance = first_attr(visit, ["insurance"])
    if insurance is not None:
        toks.append("ins_" + str(insurance))

    admission_type = first_attr(visit, ["admission_type"])
    if admission_type is not None:
        toks.append("admtype_" + str(admission_type))

    # Optional age bin if DOB exists
    dob = first_attr(patient, ["dob", "birth_datetime", "birthdate"])
    if dob is not None and admit_t is not None:
        try:
            age = int((admit_t - dob).days / 365.25)
            if age < 0:
                age = 0
            if age < 30:
                toks.append("agebin_<30")
            elif age < 50:
                toks.append("agebin_30_49")
            elif age < 70:
                toks.append("agebin_50_69")
            else:
                toks.append("agebin_>=70")
        except Exception:
            pass

    return dedup_cap(toks, 50)


def has_any_prescription(visit):
    # Label definition: any PRESCRIPTIONS record during this admission
    try:
        evs = visit.get_event_list(table="PRESCRIPTIONS")
    except Exception:
        evs = []
    return 1 if len(evs) > 0 else 0


# Task: patient -> samples
def meds_task_fn(patient):
    samples = []
    obs_delta = timedelta(days=OBS_DAYS)

    # Sort visits by admission time for stability
    visits = []
    for v in patient:
        a = visit_admit(v)
        if a is not None:
            visits.append((a, v))
    visits.sort(key=lambda x: x[0])
    visits = [v for _, v in visits]

    for visit in visits:
        admit_t = visit_admit(visit)
        disch_t = visit_discharge(visit)
        if admit_t is None or disch_t is None:
            continue

        obs_end = admit_t + obs_delta
        if disch_t < obs_end:
            obs_end = disch_t

        # If the whole stay is within obs window, "no time" events are safer
        stay_leq_obs = disch_t <= (admit_t + obs_delta)

        # Features (early window)
        lab = codes_in_window(visit, "LABEVENTS", admit_t, obs_end, allow_no_time=False)
        dx = codes_in_window(visit, "DIAGNOSES_ICD", admit_t, obs_end, allow_no_time=stay_leq_obs)
        px = codes_in_window(visit, "PROCEDURES_ICD", admit_t, obs_end, allow_no_time=stay_leq_obs)
        demo = demo_tokens(patient, visit, admit_t)

        # Label (full stay)
        label = has_any_prescription(visit)

        # Skip if truly no features at all
        if (not lab) and (not dx) and (not px) and (not demo):
            continue

        samples.append(
            {
                "patient_id": patient.patient_id,
                "visit_id": visit.visit_id,
                "labs": wrap_codes(lab, EMPTY_LAB),
                "dx": wrap_codes(dx, EMPTY_DX),
                "px": wrap_codes(px, EMPTY_PX),
                "demo": wrap_codes(demo, EMPTY_DEMO),
                "label": label,
            }
        )

    return samples


def label_stats(ds):
    # split_by_patient may return a Subset-like object, so iterate safely
    ys = [s["label"] for s in ds]
    if len(ys) == 0:
        return {"n": 0, "pos_rate": float("nan"), "pos": 0, "neg": 0}
    pos = int(sum(ys))
    neg = int(len(ys) - pos)
    return {"n": len(ys), "pos_rate": float(np.mean(ys)), "pos": pos, "neg": neg}


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_root = os.path.expanduser("~/data/mimiciii")

    # Load only needed tables (PRESCRIPTIONS needed for label existence check)
    dataset = MIMIC3Dataset(
        root=data_root,
        tables=["LABEVENTS", "DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    )

    task_dataset = dataset.set_task(meds_task_fn)

    train_ds, val_ds, test_ds = split_by_patient(task_dataset, ratios=[0.8, 0.1, 0.1], seed=seed)

    tr = label_stats(train_ds)
    va = label_stats(val_ds)
    te = label_stats(test_ds)
    print("Train:", tr)
    print("Val:", va)
    print("Test:", te)

    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)

    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds

    model = Transformer(
        dataset=base_ds,
        feature_keys=["labs", "dx", "px", "demo"],
        label_key="label",
        mode="binary",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model=model, device=device, metrics=["roc_auc", "pr_auc", "f1"])

    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=10)

    test_metrics = trainer.evaluate(test_loader)

    out_dir = os.path.join("experiments", "meds", "results")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "task": "Medication record prediction (binary)",
        "label": "has_any_prescriptions_record",
        "observation_window_days": OBS_DAYS,
        "tables_features": ["LABEVENTS", "DIAGNOSES_ICD", "PROCEDURES_ICD"],
        "table_label_source": ["PRESCRIPTIONS"],
        "model": "Transformer",
        "max_codes_per_table": MAX_CODES_PER_TABLE,
        "label_stats": {"train": tr, "val": va, "test": te},
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