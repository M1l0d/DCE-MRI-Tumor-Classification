import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nibabel as nib
from torchvision.transforms import Resize
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix, classification_report
import torchio as tio # For data augmentation

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- CONFIGURATION ---
USE_VALIDATION = True           # Set to False to evaluate on full training set
VAL_SPLIT = 0.2                 # Fraction of training set to use for validation
BATCH_SIZE = 32                  # Number of samples per batch
NUM_WORKERS = 4                 # Number of DataLoader workers
LEARNING_RATE = 0.001           # Learning rate for Adam optimizer
INPUT_SHAPE = (2, 64, 64, 64)   # Shape of input to model (channels, depth, height, width)
DROPOUT_PROB = 0.3              # Dropout probability
PATIENCE = 5                    # Patience for early stopping


# --- DATASET LOADING ---
# Define dataset paths
TIMEPOINT_DIRS = [
    "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0000",
    "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0001",
    "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0002",
]
LABELS_CSV = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/filtered_clinical_and_imaging_info_pcr.xlsx"

# Load filtered labels
labels_df = pd.read_excel(LABELS_CSV, sheet_name="Sheet1")
labels_df = labels_df.dropna(subset=["pcr"]) # Drop rows with missing pCR labels
labels_dict = dict(zip(labels_df["patient_id"], labels_df["pcr"])) # Map patient ID to pCR label
valid_patient_ids = set(labels_df["patient_id"]) # Only use patients with pCR labels

# --- AUGMENTATION PIPELINE ---
augment = tio.Compose([
    tio.RandomAffine(scales=0.05, degrees=5, translation=3, p=0.2),
    tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.1),
    tio.RandomNoise(mean=0, std=0.005, p=0.1),
])

# Custom Dataset class
class MRIDataset(Dataset):
    def __init__(self, data_dirs, labels_dict, valid_patient_ids, augment=False):
        self.data_dirs = data_dirs
        self.labels_dict = labels_dict
        self.valid_patient_ids = valid_patient_ids
        self.augment = augment
        self.file_list = [f for f in os.listdir(data_dirs[0]) if f.endswith(".nii.gz")]

        # Filter valid patients
        self.file_list = [f for f in self.file_list if "_".join(f.split("_")[:2]) in valid_patient_ids]

        # Debug: Print the file list
        print(f"Number of valid files: {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        patient_id = "_".join(file_name.split("_")[:2]) # Extract patient ID without "_0001_cropped.nii.gz"
        label = self.labels_dict.get(patient_id, 0)
        label = float(label if not pd.isna(label) else 0.0) # Get label, default to 0 if missing

        images = []
        for i, d in enumerate(self.data_dirs[:2]):  # Only use 0000 and 0001 timepoints
            timepoint_file = f"{patient_id}_{str(i).zfill(4)}_cropped.nii.gz"
            # Load NIfTI image
            image_path = os.path.join(d, timepoint_file)
            image = nib.load(image_path).get_fdata()

            # Normalize using clipped Z-score
            p2, p98 = np.percentile(image, (2, 98))
            image = np.clip(image, p2, p98)
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)
            
            # Resize image to (64, 64, 64)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Add channel dimension
            image = F.interpolate(image, size=(64, 64, 64), mode="trilinear", align_corners=False)
            image = image.squeeze(0).squeeze(0)
            images.append(image)

        # --- TEMPORAL SUBTRACTION ---
        t1 = images[1] # Timepoint 1
        delta = t1 - images[0] # Subtraction: Timepoint 1 - Timepoint 0

        image = torch.stack([t1, delta], dim=0) # Shape: (2, 64, 64, 64)

        # Apply augmentation
        if self.augment:
            image = augment(tio.Image(tensor=image, type=tio.INTENSITY)).data

        # Return image and label
        return image, torch.tensor(label, dtype=torch.float32)

"""
# Debug: Print some filenames and extracted patient IDs
print("Checking filenames and extracted patient IDs:")
for f in os.listdir(DATA_DIR)[:5]:  # Print first 5 files
    extracted_id = "_".join(f.split("_")[:2])  # Extract ID from filename
    print(f"{f} -> Extracted ID: {extracted_id}, Exists in dataset: {extracted_id in valid_patient_ids}")
"""   

# Define Dataset and DataLoader
dataset = MRIDataset(TIMEPOINT_DIRS, labels_dict, valid_patient_ids, augment=False)
# train_dataset, val_dataset = random_split(dataset, [int((1-VAL_SPLIT)*len(dataset)), int(VAL_SPLIT*len(dataset))])
# train_dataset.dataset.augment = True # Enable augmentation only for training set
# full_dataset = MRIDataset(TIMEPOINT_DIRS, labels_dict, valid_patient_ids, augment=False)

if USE_VALIDATION:
    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.augment = True # Enable augmentation only for training set

    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_counts = np.bincount([int(lbl) for lbl in train_labels])
    print("Class distribution in training set:", class_counts)

    # Calculate inverse-frequency class weights
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[int(lbl)] for lbl in train_labels]

    # Create sampler using WeightedRandomSampler for balanced minibatches
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
else:
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE , shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = train_loader

# --- MODEL DEFINITION ---
# Define 3D CNN model
class Deep3DCNN(nn.Module):
    def __init__(self):
        super(Deep3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, padding=1)  # 2 input channels now
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(2)

        # Automatically detemine the flattened size
        self.flatten_size = 128 * 4 * 4 * 4
        
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(F.relu(self.fc1(x))) # Apply dropout before fully connected layer
        x = self.fc2(x) # No Sigmoid activation because we use BCEWithLogitsLoss
        #x = self.sigmoid(self.fc2(x))
        return x
    
# --- SHALLOWER MODEL DEFINITION ---
class Shallow3DCNN(nn.Module):
    def __init__(self):
        super(Shallow3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(0.5)

        self.flatten_size = 64 * 8 * 8 * 8  # After 3x pooling on 64^3 input
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x) 

# Initialize model, loss function, and optimizer
model = Deep3DCNN().to(device)
# pos_weight = torch.tensor([1051/440], device = device) # Weight positive class more heavily since (pcr: 0=1051 and 1=440)
pos_weight = torch.tensor([2.0], device=device) # Weight positive class more heavily since (pcr: 0=1051 and 1=440)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# --- EVALUATION ---
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.squeeze().cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)

    print("\n--- Evaluation Metrics ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, y_prob))
    print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred))

# --- TRAINING ---

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=15):
    model.train()
    train_losses = []
    val_losses = []

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_multichannel_aug.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        with torch.no_grad():
            model.eval()
            all_val_labels = []
            all_val_probs = []
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs.squeeze().cpu().numpy())
            auc_score = roc_auc_score(all_val_labels, all_val_probs)
            print(f"Validation ROC AUC: {auc_score:.4f}")
            
    # Plot training and validation loss
    plt.figure()
    epochs_ran = len(train_losses)
    plt.plot(range(1, epochs_ran + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs_ran + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/loss_curve_multichannel_aug.png")
    print("Loss curve saved as loss_curve.png")

# Train and evaluate the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=30)
model.load_state_dict(torch.load("best_model_multichannel_aug.pt"))
evaluate_model(model, val_loader)
print("✅ Training complete!")