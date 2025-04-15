import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

@torch.no_grad()
def evaluate_model(model, val_loader, model_path, device, run_dir):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_logits, all_labels = [], []

    for batch in val_loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        logits = model(x).squeeze()
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)

    print(f"🔍 Predicted probabilities: mean={probs.mean():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")
    print(f"📊 Class distribution in val set: {Counter(labels.numpy())}")

    try:
        auc = roc_auc_score(labels.numpy(), probs.numpy())
    except ValueError:
        auc = float('nan')
    print(f"AUC: {auc:.4f}")

    best_f1 = 0
    best_thresh = 0.5
    print("\n📊 Threshold sweep:")
    print(f"{'Thresh':>7} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}")
    print("-" * 40)
    for t in np.arange(0.1, 0.91, 0.05):
        preds = (probs >= t).int()
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        print(f"{t:7.2f} | {acc:6.3f} | {prec:6.3f} | {rec:6.3f} | {f1:6.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"\n🎯 Best Threshold: {best_thresh:.2f} (F1={best_f1:.3f})")
    final_preds = (probs >= best_thresh).int()
    acc = accuracy_score(labels, final_preds)
    prec = precision_score(labels, final_preds, zero_division=0)
    rec = recall_score(labels, final_preds, zero_division=0)
    f1 = f1_score(labels, final_preds, zero_division=0)
    cm = confusion_matrix(labels, final_preds)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Best Threshold: {best_thresh:.2f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")