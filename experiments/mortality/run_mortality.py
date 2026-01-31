import os, json, random
import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import mortality_prediction_mimic3_fn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def main():
    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load MIMIC-III dataset
    dataset = MIMIC3Dataset(
        root=os.path.expanduser("~/data/mimiciii"),
        tables=["ADMISSIONS", "DIAGNOSES_ICD", "LABEVENTS", "PRESCRIPTIONS"],
    )

    # Define in-hospital mortality prediction task (first 24h)
    task_dataset = dataset.set_task(
        mortality_prediction_mimic3_fn,
        prediction_window=24 * 60,
    )

    # Train / val / test split
    n = len(task_dataset)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # Extract string-based medical codes as tokens
    def extract_tokens(sample):
        tokens = []
        for k, v in sample.items():
            if k == "label":
                continue
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        tokens.append(f"{k}:{item}")
        return tokens

    # Build vocabulary from training set only
    vocab = {}
    for i in train_idx:
        for t in extract_tokens(task_dataset[i]):
            if t not in vocab:
                vocab[t] = len(vocab)
    input_dim = len(vocab)

    # Convert one sample to multi-hot vector + label
    def featurize(sample):
        x = np.zeros(input_dim, dtype=np.float32)
        for t in extract_tokens(sample):
            j = vocab.get(t)
            if j is not None:
                x[j] = 1.0
        y = float(sample["label"])
        return x, y

    # Build numpy datasets
    def build_dataset(indices):
        X, Y = [], []
        for i in indices:
            x, y = featurize(task_dataset[i])
            X.append(x)
            Y.append(y)
        return np.stack(X), np.asarray(Y, dtype=np.float32)

    X_train, y_train = build_dataset(train_idx)
    X_val, y_val = build_dataset(val_idx)
    X_test, y_test = build_dataset(test_idx)

    # Logistic regression model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(input_dim, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    batch_size = 256
    epochs = 5

    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_train))
        for start in range(0, len(X_train), batch_size):
            b = perm[start:start + batch_size]
            xb = torch.tensor(X_train[b], device=device)
            yb = torch.tensor(y_train[b], device=device)
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(X_test, device=device)
        ).squeeze(1).cpu().numpy()

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    f1 = f1_score(y_test, preds)

    # Save metrics
    out_dir = os.path.join("experiments", "mortality", "results")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "metrics": {"auc": float(auc), "auprc": float(auprc), "f1": float(f1)},
        "threshold": 0.5,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("AUC:", auc, "AUPRC:", auprc, "F1:", f1)


if __name__ == "__main__":
    main()