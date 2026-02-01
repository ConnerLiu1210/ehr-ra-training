# run_mortality.py
# Baseline: In-hospital Mortality (MIMIC-III)
# Model: PyHealth built-in Transformer
# Metrics: ROC-AUC, PR-AUC (AUPRC), F1

import os
import json
import random
import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.tasks import mortality_prediction_mimic3_fn
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer


def main():
    # reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_root = os.path.expanduser("~/data/mimiciii")

    # Observation window in minutes (e.g., 48h)
    prediction_window = 48 * 60

    # Load raw MIMIC-III tables
    dataset = MIMIC3Dataset(
        root=data_root,
        tables=[
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
        ],
    )

    # Convert raw tables into task samples
    task_dataset = dataset.set_task(
        mortality_prediction_mimic3_fn,
    )

    # Split
    train_ds, val_ds, test_ds = split_by_patient(
        task_dataset,
        ratios=[0.8, 0.1, 0.1],
        seed=seed,
    )

    # Build dataloaders for training/evaluation
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)

    # Initialize Transformer model for dataset
    model = Transformer(dataset=train_ds)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training loop & metric computation
    trainer = Trainer(
        model=model,
        device=device,
        metrics=["roc_auc", "pr_auc", "f1"],
    )

    # Train the model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
    )

    # Evaluate on test split
    test_metrics = trainer.evaluate(test_loader)

    # Save metrics to a JSON file
    out_dir = os.path.join("experiments", "mortality", "results")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "task": "in-hospital mortality (mimic-iii)",
        "model": "Transformer",
        "prediction_window_minutes": int(prediction_window),
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