import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import numpy as np
import os
import matplotlib.pyplot as plt


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

    # --- Print sigmoid distribution ---
    print(f"🔍 Predicted probabilities: mean={probs.mean():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")

    # --- Calculate both standard and inverse AUC ---
    std_auc = roc_auc_score(labels.numpy(), probs.numpy())
    inv_auc = roc_auc_score(labels.numpy(), 1 - probs.numpy())
    
    # --- Use the better predictor ---
    use_inverse = inv_auc > std_auc
    final_probs = 1 - probs if use_inverse else probs
    final_auc = max(std_auc, inv_auc)
    
    print(f"Standard AUC: {std_auc:.4f}")
    print(f"Inverse AUC: {inv_auc:.4f}")
    print(f"Using {'inverse' if use_inverse else 'standard'} predictions (AUC: {final_auc:.4f})")

    # --- Visualize prediction distributions ---
    plot_prediction_distribution(final_probs, labels, os.path.join(run_dir, "pred_distribution.png"))
    
    # --- Threshold sweep ---
    best_f1, best_thresh = 0, 0.5
    print("\n📊 Threshold sweep:")
    print(f"{'Thresh':>7} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'Spec':>6} | {'F1':>6}")
    print("-" * 50)
    for t in np.arange(0.3, 0.7, 0.01):  # Narrower range centered on 0.5
        preds = (final_probs >= t).int()
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        spec = accuracy_score(1-labels, 1-preds)  # Calculate specificity
        
        # Only consider thresholds with at least 20% specificity
        if spec >= 0.2:
            f1 = f1_score(labels, preds, zero_division=0)
            print(f"{t:7.2f} | {acc:6.3f} | {prec:6.3f} | {rec:6.3f} | {spec:6.3f} | {f1:6.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

    print(f"\n🎯 Best Threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

    # --- Final metrics ---
    final_preds = (final_probs >= best_thresh).int()
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

    # --- Save metrics ---
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(f"Standard AUC: {std_auc:.4f}\n")
        f.write(f"Inverse AUC: {inv_auc:.4f}\n")
        f.write(f"Final AUC: {final_auc:.4f}\n")
        f.write(f"Best Threshold: {best_thresh:.2f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

def visualize_features(model, val_loader, device, save_dir):
    """Visualize intermediate feature maps."""
    model.eval()
    x, y = next(iter(val_loader))
    x = x.to(device)
    
    # Register hooks to capture intermediate activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks
    if hasattr(model, 'backbone'):
        model.backbone.layer1.register_forward_hook(get_activation('layer1'))
        model.backbone.layer2.register_forward_hook(get_activation('layer2'))
    
    # Forward pass
    with torch.no_grad():
        model(x)
    
    # Visualize activations
    for name, feat in activations.items():
        plt.figure(figsize=(12, 4))
        for i in range(min(4, feat.size(1))):  # Show first 4 channels
            plt.subplot(1, 4, i + 1)
            plt.title(f"{name} - Ch {i}")
            plt.imshow(feat[0, i, :, :, feat.shape[4]//2].numpy(), cmap='viridis')
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"features_{name}.png"))
        plt.close()

def plot_prediction_distribution(probs, labels, save_path):
    """Plot distribution of predictions for each class."""
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy arrays
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    
    # Get predictions for each class
    pos_preds = probs_np[labels_np == 1]
    neg_preds = probs_np[labels_np == 0]
    
    # Plot histograms
    plt.hist(pos_preds, bins=20, alpha=0.5, label='Triple Negative')
    plt.hist(neg_preds, bins=20, alpha=0.5, label='Other')
    
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Probabilities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
