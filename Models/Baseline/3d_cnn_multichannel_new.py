import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import torchio as tio

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

VAL_SPLIT = 0.2
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
PATIENCE = 6
MAX_EPOCHS = 50
DROPOUT_PROB = 0.3
INPUT_TYPE = "t0_t1_t2"  # options: 't0_t1_t2', 't1_t2', 't2_only'

# --- PATHS ---
TIMEPOINT_DIRS = [
    "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0000",
    "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0001",
    "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0002"
]
LABELS_CSV = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/filtered_clinical_and_imaging_info_pcr.xlsx"


# --- DATASET ---
class MRIDataset(Dataset):
    def __init__(self, data_dirs, labels_dict, valid_ids, augment=False, input_type="t0_t1_t2"):
        self.data_dirs = data_dirs
        self.labels_dict = labels_dict
        self.valid_ids = valid_ids
        self.augment = augment
        self.input_type = input_type
        self.files = [f for f in os.listdir(data_dirs[0]) if "_".join(f.split("_")[:2]) in valid_ids]

        self.transforms = tio.Compose([
            tio.RandomAffine(scales=0.05, degrees=5, translation=3, p=0.2),
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.1),
            tio.RandomNoise(mean=0, std=0.005, p=0.1)
        ])

        print(f"✅ Loaded {len(self.files)} valid samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        pid = "_".join(filename.split("_")[:2])
        label = float(self.labels_dict.get(pid, 0.0))

        timepoints = []
        for i, path in enumerate(self.data_dirs):
            nii = nib.load(os.path.join(path, f"{pid}_{str(i).zfill(4)}_cropped.nii.gz"))
            vol = nii.get_fdata()
            p2, p98 = np.percentile(vol, (2, 98))
            vol = np.clip(vol, p2, p98)
            vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-5)
            vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            vol_tensor = F.interpolate(vol_tensor, size=(64, 64, 64), mode='trilinear', align_corners=False)
            timepoints.append(vol_tensor.squeeze())

        # Log some stats
        if idx < 3:
            print(f"[DEBUG] t0: μ={timepoints[0].mean():.4f}, σ={timepoints[0].std():.4f}")
            print(f"[DEBUG] t1: μ={timepoints[1].mean():.4f}, σ={timepoints[1].std():.4f}")
            print(f"[DEBUG] t2: μ={timepoints[2].mean():.4f}, σ={timepoints[2].std():.4f}")

        # Input selection
        if self.input_type == "t0_t1_t2":
            input_tensor = torch.stack([timepoints[0], timepoints[1], timepoints[2]], dim=0)
        elif self.input_type == "t1_t2":
            input_tensor = torch.stack([timepoints[1], timepoints[2]], dim=0)
        elif self.input_type == "t2_only":
            input_tensor = timepoints[2].unsqueeze(0)
        else:
            raise ValueError(f"Invalid input_type: {self.input_type}")

        if self.augment:
            input_tensor = self.transforms(tio.Image(tensor=input_tensor, type=tio.INTENSITY)).data

        return input_tensor, torch.tensor(label, dtype=torch.float32)


# --- MODEL ---
class Deep3DCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.InstanceNorm3d(64)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.InstanceNorm3d(128)

        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# --- TRAINING ---
def train_model(model, train_loader, val_loader, criterion, optimizer):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_auc = 0
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, val_labels, val_probs = 0, [], []
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

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # Plot loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("📈 Saved loss curve.")


# --- EVALUATION ---
def evaluate_model(model, loader):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x))
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    y_true = np.array(all_labels)
    y_scores = np.array(all_probs)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    y_pred = (y_scores > best_thresh).astype(int)

    print(f"\n🎯 Optimal Threshold based on F1: {best_thresh:.4f}")
    print("\n--- Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_scores[best_idx]:.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_true, y_scores):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


# --- MAIN ---
labels_df = pd.read_excel(LABELS_CSV)
labels_df = labels_df.dropna(subset=["pcr"])
labels_dict = dict(zip(labels_df["patient_id"], labels_df["pcr"]))
valid_ids = set(labels_df["patient_id"])

dataset = MRIDataset(TIMEPOINT_DIRS, labels_dict, valid_ids, augment=True, input_type=INPUT_TYPE)
val_size = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
class_counts = np.bincount([int(lbl) for lbl in train_labels])
weights = 1.0 / class_counts
sample_weights = [weights[int(lbl)] for lbl in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = Deep3DCNN(in_channels={"t2_only": 1, "t1_t2": 2, "t0_t1_t2": 3}[INPUT_TYPE]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

train_model(model, train_loader, val_loader, criterion, optimizer)
model.load_state_dict(torch.load("best_model.pt"))
evaluate_model(model, val_loader)
