import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def validate(model, val_loader, device, criterion, threshold=0.5):
    model.eval()
    val_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Binarize predictions with custom threshold
    binary_preds = (all_preds > threshold).float()

    metrics = {
        'auc': roc_auc_score(all_labels.numpy(), all_preds.numpy()),
        'accuracy': accuracy_score(all_labels.numpy(), binary_preds.numpy()),
        'precision': precision_score(all_labels.numpy(), binary_preds.numpy(), zero_division=0),
        'recall': recall_score(all_labels.numpy(), binary_preds.numpy(), zero_division=0),
        'f1': f1_score(all_labels.numpy(), binary_preds.numpy(), zero_division=0)
    }

    avg_val_loss = val_loss / len(val_loader.dataset)

    return avg_val_loss, metrics

def evaluate_model(model, val_loader, device, threshold=0.35):
    """Wrapper function for model evaluation with custom threshold"""
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss, metrics = validate(model, val_loader, device, criterion, threshold=threshold)
    
    print("\n| Metric    | Value     |")
    print("|-----------|-----------|")
    print(f"| Loss      | {val_loss:.6f} |")
    for metric, value in metrics.items():
        print(f"| {metric.capitalize():9s} | {value:.6f} |")
    
    return metrics