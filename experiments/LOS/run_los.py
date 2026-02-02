# Task: LOS prediction (binary classification)
# Observation: first OBS_DAYS after admission (capped at discharge)
# Tables: LABEVENTS + PRESCRIPTIONS + DIAGNOSES_ICD + PROCEDURES_ICD + CHARTEVENTS
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

OBS_DAYS = 2
LOS_THRESHOLD_DAYS = 6

# Keep sequences from exploding
MAX_CODES_PER_TABLE = 200

# Placeholder tokens to avoid empty features (prevents PyHealth validation errors)
EMPTY_LAB = "__EMPTY_LAB__"
EMPTY_RX = "__EMPTY_RX__"
EMPTY_DX = "__EMPTY_DX__"
EMPTY_PX = "__EMPTY_PX__"
EMPTY_CHART = "__EMPTY_CHART__"
EMPTY_DEMO = "__EMPTY_DEMO__"
EMPTY_ICU = "__EMPTY_ICU__"


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


# Visit / event field accessors
def get_visit_admit_time(visit):
    return get_first_attr(visit, ["encounter_time", "start_time", "admit_time", "admittime"])


def get_visit_discharge_time(visit):
    return get_first_attr(visit, ["discharge_time", "end_time", "dischtime", "discharge_datetime"])


def get_event_time(ev):
    # Common timestamp fields across different event objects
    return get_first_attr(ev, ["timestamp", "charttime", "time", "event_time", "datetime"])


def get_event_code(ev):
    # Common code fields for labs, meds, diagnoses, procedures, charts
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


def build_demo_tokens(patient, visit, admit):
    # Build a small set of categorical tokens from patient/admission info
    tokens = []

    gender = get_first_attr(patient, ["gender", "sex"])
    if gender is not None:
        tokens.append("gender_" + str(gender))

    ethnicity = get_first_attr(visit, ["ethnicity"])
    if ethnicity is not None:
        tokens.append("eth_" + str(ethnicity))

    insurance = get_first_attr(visit, ["insurance"])
    if insurance is not None:
        tokens.append("ins_" + str(insurance))

    admission_type = get_first_attr(visit, ["admission_type"])
    if admission_type is not None:
        tokens.append("admtype_" + str(admission_type))

    dob = get_first_attr(patient, ["dob", "birth_datetime", "birthdate"])
    if dob is not None and admit is not None:
        try:
            age = int((admit - dob).days / 365.25)
            if age < 0:
                age = 0
            if age < 30:
                tokens.append("agebin_<30")
            elif age < 50:
                tokens.append("agebin_30_49")
            elif age < 70:
                tokens.append("agebin_50_69")
            else:
                tokens.append("agebin_>=70")
        except Exception:
            pass

    return tokens


def build_icu_tokens(visit):
    # ICU-related categorical tokens if available
    tokens = []

    first_careunit = get_first_attr(visit, ["first_careunit", "first_care_unit"])
    if first_careunit is not None:
        tokens.append("icu_first_" + str(first_careunit))

    last_careunit = get_first_attr(visit, ["last_careunit", "last_care_unit"])
    if last_careunit is not None:
        tokens.append("icu_last_" + str(last_careunit))

    los = get_first_attr(visit, ["los", "icu_los"])
    if los is not None:
        try:
            los = float(los)
            if los < 1:
                tokens.append("iculos_<1d")
            elif los < 3:
                tokens.append("iculos_1_3d")
            elif los < 7:
                tokens.append("iculos_3_7d")
            else:
                tokens.append("iculos_>=7d")
        except Exception:
            pass

    return tokens


def get_samples_list(ds):
    # split_by_patient often returns torch.utils.data.Subset
    if hasattr(ds, "samples"):
        return ds.samples
    if hasattr(ds, "dataset") and hasattr(ds, "indices") and hasattr(ds.dataset, "samples"):
        return [ds.dataset.samples[i] for i in ds.indices]
    try:
        return [ds[i] for i in range(len(ds))]
    except Exception:
        return []


def los_long_fn(patient):
    # Build samples for ONE patient
    # x: codes from multiple tables within first OBS_DAYS after admission (capped by discharge)
    # y: 1 if LOS >= LOS_THRESHOLD_DAYS else 0

    samples = []
    obs_window = timedelta(days=OBS_DAYS)

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

        try:
            los_days = (discharge - admit).total_seconds() / 86400.0
        except Exception:
            continue

        label = 1 if los_days >= float(LOS_THRESHOLD_DAYS) else 0

        # Observation end = min(admit + OBS_DAYS, discharge)
        obs_end = admit + obs_window
        if discharge < obs_end:
            obs_end = discharge

        # If the whole stay is within obs window, allow "no timestamp" items
        stay_leq_obs = discharge <= (admit + obs_window)

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
        rx_codes = filter_codes_in_window(rx_evs, admit, obs_end, allow_no_time=stay_leq_obs)

        # DIAGNOSES_ICD
        try:
            dx_evs = visit.get_event_list(table="DIAGNOSES_ICD")
        except Exception:
            dx_evs = []
        dx_codes = filter_codes_in_window(dx_evs, admit, obs_end, allow_no_time=stay_leq_obs)

        # PROCEDURES_ICD
        try:
            px_evs = visit.get_event_list(table="PROCEDURES_ICD")
        except Exception:
            px_evs = []
        px_codes = filter_codes_in_window(px_evs, admit, obs_end, allow_no_time=stay_leq_obs)

        # CHARTEVENTS
        try:
            chart_evs = visit.get_event_list(table="CHARTEVENTS")
        except Exception:
            chart_evs = []
        chart_codes = filter_codes_in_window(chart_evs, admit, obs_end, allow_no_time=False)

        # Extra categorical tokens
        demo_tokens = dedup_and_cap(build_demo_tokens(patient, visit, admit), 50)
        icu_tokens = dedup_and_cap(build_icu_tokens(visit), 50)

        # Skip if everything is missing
        if (
            len(lab_codes) == 0
            and len(rx_codes) == 0
            and len(dx_codes) == 0
            and len(px_codes) == 0
            and len(chart_codes) == 0
            and len(demo_tokens) == 0
            and len(icu_tokens) == 0
        ):
            continue

        samples.append(
            {
                "patient_id": patient.patient_id,
                "visit_id": visit.visit_id,
                "labs": wrap_feature(lab_codes, EMPTY_LAB),
                "rx": wrap_feature(rx_codes, EMPTY_RX),
                "dx": wrap_feature(dx_codes, EMPTY_DX),
                "px": wrap_feature(px_codes, EMPTY_PX),
                "chart": wrap_feature(chart_codes, EMPTY_CHART),
                "demo": wrap_feature(demo_tokens, EMPTY_DEMO),
                "icu": wrap_feature(icu_tokens, EMPTY_ICU),
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

    dataset = MIMIC3Dataset(
        root=data_root,
        tables=["LABEVENTS", "PRESCRIPTIONS", "DIAGNOSES_ICD", "PROCEDURES_ICD", "CHARTEVENTS"],
    )

    task_dataset = dataset.set_task(los_long_fn)

    train_ds, val_ds, test_ds = split_by_patient(task_dataset, ratios=[0.8, 0.1, 0.1], seed=seed)

    # Label prevalence check (Subset-safe)
    train_samples = get_samples_list(train_ds)
    y_train = [s["label"] for s in train_samples if isinstance(s, dict) and "label" in s]
    pos_rate = float(np.mean(y_train)) if len(y_train) > 0 else float("nan")
    print("Train samples:", len(train_samples), "Pos rate:", pos_rate)

    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)

    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds

    model = Transformer(
        dataset=base_ds,
        feature_keys=["labs", "rx", "dx", "px", "chart", "demo", "icu"],
        label_key="label",
        mode="binary",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(model=model, device=device, metrics=["roc_auc", "pr_auc", "f1"])

    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=10)

    test_metrics = trainer.evaluate(test_loader)

    out_dir = os.path.join("experiments", "los", "results")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "task": "LOS long stay classification",
        "model": "Transformer",
        "obs_days": OBS_DAYS,
        "los_threshold_days": LOS_THRESHOLD_DAYS,
        "max_codes_per_table": MAX_CODES_PER_TABLE,
        "tables": ["LABEVENTS", "PRESCRIPTIONS", "DIAGNOSES_ICD", "PROCEDURES_ICD", "CHARTEVENTS"],
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