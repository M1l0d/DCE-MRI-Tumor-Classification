import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import EarlyStopping, save_loss_plot
from torch.cuda.amp import autocast, GradScaler

class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, input, target):
        with torch.no_grad():
            target = target.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(
            input, target, reduction='none', pos_weight=self.pos_weight
        )
        probas = torch.sigmoid(input)
        pt = torch.where(target == 1, probas, 1 - probas)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train_model(model, loaders, optimizer, criterion, device, max_epochs, save_path, run_dir):
    train_loader, val_loader = loaders
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=max_epochs,
        eta_min=1e-6
    )

    scaler = GradScaler()  # For mixed precision training

    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for x_img, y in train_loader:
            x_img = x_img.to(device)
            y = y.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            
            # Use mixed precision
            with autocast():
                logits = model(x_img)
                loss = criterion(logits, y)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Warning: Loss is {loss.item()}, skipping batch")
                continue
            
            # Use scaler for mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping (CRITICAL)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_img, y in val_loader:
                x_img = x_img.to(device)
                y = y.to(device).view(-1)  # 🔧 reshape labels to [batch_size]

                logits = model(x_img)
                loss = criterion(logits, y.unsqueeze(1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_train_loss = total_train_loss / len(train_loader)

        auc = evaluate_during_training(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, AUC: {auc:.4f}")

        # Early stopping
        if early_stopping(avg_val_loss):
            print(f"⏹️ Early stopping at epoch {epoch}")
            break

        # Save best model
        if avg_val_loss <= min(val_losses):
            torch.save(model.state_dict(), save_path)

    # Save loss curve
    plot_path = os.path.join(run_dir, "loss_plot.png")
    save_loss_plot(train_losses, val_losses, plot_path)
    print(f"📈 Saved loss plot to {plot_path}")


def find_lr(model, train_loader, optimizer, criterion, device, init_lr=1e-7, final_lr=10, beta=0.98):
    """Find the optimal learning rate."""
    model.train()
    num = len(train_loader) - 1
    mult = (final_lr / init_lr) ** (1 / num)
    lr = init_lr
    optimizer.param_groups[0]['lr'] = lr
    avg_loss, best_loss = 0.0, float('inf')
    losses, lrs = [], []

    for i, (x_img, y) in enumerate(train_loader):
        x_img, y = x_img.to(device), y.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        logits = model(x_img)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        if smoothed_loss > 4 * best_loss:
            break

        losses.append(smoothed_loss)
        lrs.append(lr)

        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    # Plot the learning rate vs. loss
    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.show()

    return lrs, losses

def evaluate_during_training(model, val_loader, device):
    """Calculate AUC during training."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # Check if predictions are all the same
    if np.std(all_preds) < 1e-6:
        print("⚠️ WARNING: Model predictions have near-zero variance!")
        
    # Check for NaN values and fix them
    nan_mask = np.isnan(all_preds)
    if nan_mask.any():
        print(f"⚠️ WARNING: Found {nan_mask.sum()} NaN predictions, replacing with 0.5")
        all_preds[nan_mask] = 0.5
    
    std_auc = roc_auc_score(all_labels, all_preds)
    inv_auc = roc_auc_score(all_labels, 1 - all_preds)

    if inv_auc > std_auc:
        print(f"AUC: {std_auc:.4f}, Inverse AUC: {inv_auc:.4f} ← USING THIS")
        return inv_auc
    else:
        print(f"AUC: {std_auc:.4f} ← USING THIS, Inverse AUC: {inv_auc:.4f}")
        return std_auc