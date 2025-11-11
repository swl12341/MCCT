# evaluate.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


def select_peak_predictions(probs, preds):
    probs = np.asarray(probs).reshape(-1)
    preds = np.asarray(preds).reshape(-1)

    filtered = np.zeros_like(preds)
    start = None

    for idx, label in enumerate(preds):
        if label == 1:
            if start is None:
                start = idx
        else:
            if start is not None:
                segment = slice(start, idx)
                best_idx = segment.start + np.argmax(probs[segment])
                filtered[best_idx] = 1
                start = None

    if start is not None:
        segment = slice(start, len(preds))
        best_idx = segment.start + np.argmax(probs[segment])
        filtered[best_idx] = 1

    return filtered

def balance_validation_data(X, y):
    X = np.array(X)
    y = np.array(y)
    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]
    n_samples = min(len(class0_idx), len(class1_idx))
    idx0 = np.random.choice(class0_idx, n_samples, replace=False)
    idx1 = np.random.choice(class1_idx, n_samples, replace=False)
    combined_idx = np.concatenate([idx0, idx1])
    np.random.shuffle(combined_idx)
    return X[combined_idx], y[combined_idx]


def evaluate_model(model,
                   data_loader,
                   device,
                   output_dir=None,
                   is_best=False,
                   verbose=False,
                   tag="val",
                   balance=False,
                   return_details=False,
                   enforce_peak=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)  # [B]
            probs = torch.sigmoid(outputs).cpu().numpy()  # sigmoid概率
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds).reshape(-1)
    all_labels = np.array(all_labels).reshape(-1)
    all_probs = np.array(all_probs).reshape(-1)

    if balance:
        features = np.stack([all_probs, all_preds], axis=1)
        X_bal, y_bal = balance_validation_data(features, all_labels)
        all_probs = X_bal[:, 0]
        all_preds = X_bal[:, 1].astype(int)
        all_labels = y_bal

    peak_preds = select_peak_predictions(all_probs, all_preds)
    active_preds = peak_preds if enforce_peak else all_preds

    cm = confusion_matrix(all_labels, active_preds)
    if verbose:
        print(classification_report(all_labels, active_preds, digits=4))
    auc = roc_auc_score(all_labels, all_probs)
    acc = np.mean(active_preds == all_labels)

    if is_best and output_dir is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (Best {tag})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_best_{tag}.png"), dpi=600)
        plt.close()

    if return_details:
        details = {
            "labels": all_labels,
            "preds": all_preds,
            "probs": all_probs,
            "peak_preds": peak_preds,
            "final_preds": active_preds
        }
        return acc, auc, details

    return acc, auc
