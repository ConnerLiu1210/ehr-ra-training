# run_mortality.py
# baseline for in-hospital mortality prediction (MIMIC-III)
# Model: Logistic Regression
# Level: admission-level
# Metrics: AUC, AUPRC, F1

import os, json, random
import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import MortalityPrediction
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def main():
    # 0. Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load MIMIC-III data
    dataset = MIMIC3Dataset(
        root=os.path.expanduser("~/data/mimiciii"),
        tables=[
            "ADMISSIONS",
            "DIAGNOSES_ICD",
            "LABEVENTS",
            "PRESCRIPTIONS",
        ],
    )

    # Define the task: in-hospital mortality prediction
    # Use the first 24 hours after admission as observation window
    task = MortalityPrediction(
        dataset=dataset,
        prediction_window=24 * 60,  # minutes
    )

    # 2. Train / Val / Test split
    n = len(task)
    idx = list(range(n))
    random.shuffle(idx)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # 3. Extract features and labels
    def get_xy(i):
        sample = task[i]
        x = np.asarray(sample["x"], dtype=np.float32)
        y = int(sample["label"])
        return x, y

    # Infer feature dimension from one sample
    x0, _ = get_xy(train_idx[0])
    input_dim = x0.shape[0]

    def build_dataset(indices):
        X, Y = [], []
        for i in indices:
            x, y = get_xy(i)
            X.append(x)
            Y.append(y)
        return np.stack(X), np.asarray(Y, dtype=np.float32)

    X_train, y_train = build_dataset(train_idx)
    X_val, y_val = build_dataset(val_idx)
    X_test, y_test = build_dataset(test_idx)

    # 4. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5. Model definition
    # Logistic Regression: a single linear layer
    # Output is a logit (before sigmoid)
    model = nn.Linear(input_dim, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training loop
    batch_size = 256
    epochs = 5

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_train))

        for start in range(0, len(X_train), batch_size):
            idx_batch = perm[start:start + batch_size]

            xb = torch.tensor(X_train[idx_batch], device=device)
            yb = torch.tensor(y_train[idx_batch], device=device)

            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 7. Evaluation on test set
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(X_test, device=device)
        ).squeeze(1).cpu().numpy()

    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    f1 = f1_score(y_test, preds)

    # 8. Save results
    out_dir = "experiments/mortality/results"
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "task": "in-hospital mortality (mimic-iii)",
        "prediction_window_minutes": 24 * 60,
        "model": "logistic_regression",
        "metrics": {
            "auc": float(auc),
            "auprc": float(auprc),
            "f1": float(f1),
        },
        "threshold": 0.5,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("AUC:", auc, "AUPRC:", auprc, "F1:", f1)
    print("Saved results to:", os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()