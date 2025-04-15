import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch
import json

def evaluate_model(model, loader, device, save_path=None):
    model.eval()
    y_true, y_scores = [], []

    with torch.no_grad():
        for x_img, x_tabular, y in loader:
            x_img = x_img.to(device)
            x_tabular = x_tabular.to(device)
            logits = model(x_img, x_tabular)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_scores.extend(probs)
            y_true.extend(y.numpy().flatten())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    y_pred = (y_scores > best_thresh).astype(int)

    print(f"🎯 Threshold: {best_thresh:.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_scores):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1s[best_idx]:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    if save_path:
        with open(save_path, "w") as f:
            json.dump({
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred)),
                "recall": float(recall_score(y_true, y_pred)),
                "f1_score": float(f1s[best_idx]),
                "roc_auc": float(roc_auc_score(y_true, y_scores)),
                "threshold": float(best_thresh),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }, f, indent=2)
        print(f"📊 Metrics saved to {save_path}")