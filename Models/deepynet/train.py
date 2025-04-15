import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def train_model(model, loaders, optimizer, criterion, device, max_epochs=50, patience=6, save_path="best_model.pt"):
    train_loader, val_loader = loaders
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_auc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).view(-1, 1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_labels, val_probs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).view(-1, 1)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels.extend(y.cpu().numpy().flatten())

        val_losses.append(val_loss / len(val_loader))
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, AUC: {val_auc:.4f}")

        # --- Early Stopping ---
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # --- Plot Losses ---
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path.replace(".pt", "_loss.png"))
    print(f"📈 Saved loss plot to {save_path.replace('.pt', '_loss.png')}")