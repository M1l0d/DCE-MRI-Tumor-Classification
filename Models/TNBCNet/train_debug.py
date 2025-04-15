import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import EarlyStopping, save_loss_plot

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

def train_model(model, loaders, optimizer, criterion, device, max_epochs, save_path, run_dir, debug_overfit=False):
    train_loader, val_loader = loaders
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    train_losses, val_losses = [], []

    if debug_overfit:
        # Grab a single batch from the training loader and repeat it
        fixed_batch = next(iter(train_loader))
        x_img, y = fixed_batch
        x_img, y = x_img.to(device), y.to(device).float().view(-1, 1)
        print(f"🔬 Debug batch shape: {x_img.shape}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0

        if debug_overfit:
            optimizer.zero_grad()
            logits = model(x_img)
            loss = criterion(logits, y.view(-1, 1))
            loss.backward()
            optimizer.step()
            avg_train_loss = loss.item()
        else:
            for x_img, y in train_loader:
                x_img = x_img.to(device)
                y = y.to(device).float().view(-1, 1)
                optimizer.zero_grad()
                logits = model(x_img)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)

        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_img, y in val_loader:
                x_img = x_img.to(device)
                y = y.to(device).view(-1)
                logits = model(x_img)
                loss = criterion(logits, y.unsqueeze(1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if early_stopping(avg_val_loss):
            print(f"⏹️ Early stopping at epoch {epoch}")
            break

        if avg_val_loss <= min(val_losses):
            torch.save(model.state_dict(), save_path)

    plot_path = os.path.join(run_dir, "loss_plot.png")
    save_loss_plot(train_losses, val_losses, plot_path)
    print(f"📈 Saved loss plot to {plot_path}")
