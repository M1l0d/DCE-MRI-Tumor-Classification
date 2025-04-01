import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix, classification_report

# --- CONFIGURATION ---
USE_VALIDATION = True   # Set to False to evaluate on full training set
VAL_SPLIT = 0.2         # Fraction of training set to use for validation

# --- DATASET LOADING ---
# Define dataset paths
DATA_DIR = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0001"
LABELS_CSV = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/filtered_clinical_and_imaging_info_pcr.xlsx"

# Load filtered labels
labels_df = pd.read_excel(LABELS_CSV, sheet_name="Sheet1")
labels_df = labels_df.dropna(subset=["pcr"]) # Drop rows with missing pCR labels
labels_dict = dict(zip(labels_df["patient_id"], labels_df["pcr"])) # Map patient ID to pCR label
valid_patient_ids = set(labels_df["patient_id"]) # Only use patients with pCR labels

# Custom Dataset class
class MRIDataset(Dataset):
    def __init__(self, data_dir, labels_dict, valid_patient_ids, transform=None):
        self.data_dir = data_dir
        self.labels_dict = labels_dict
        self.valid_patient_ids = valid_patient_ids
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".nii.gz")]
        
        # Extract patient ID correctly by removing "_0001_cropped.nii.gz"
        self.file_list = [f for f in self.file_list if "_".join(f.split("_")[:2]) in valid_patient_ids] # Filter valid patients
        self.transform = transform

        # Debug: Print the file list
        #print(f"Filtered file list: {self.file_list}")
        print(f"Number of valid files: {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        #patient_id = file_name.split("_")[0] # Extract patient ID
        patient_id = "_".join(file_name.split("_")[:2]) # Extract patient ID without "_0001_cropped.nii.gz"
        label = self.labels_dict.get(patient_id, 0) # Get label, default to 0 if missing
        if pd.isna(label):
            label = 0.0
        label = float(label) # Convert to float

        # Load NIfTI image
        image_path = os.path.join(self.data_dir, file_name)
        image = nib.load(image_path).get_fdata()

        # Normalize intensity (Z-score)
        image = (image - np.mean(image)) / np.std(image)

        # Resize image to (64, 64, 64)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # Add channel dimension
        image = F.interpolate(image.unsqueeze(0), size=(64, 64, 64), mode="trilinear", align_corners=False).squeeze(0)

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
dataset = MRIDataset(DATA_DIR, labels_dict, valid_patient_ids)

if USE_VALIDATION:
    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
else:
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = train_loader

# Define 3D CNN model
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(16)

        # Automatically detemine the flattened size
        self.flatten_size = None # This will be set dynamically    
        
        self.fc1 = nn.Linear(1, 1) # Placeholder, will be set dynamically in forward()
        self.fc2 = nn.Linear(128, 1)
        #self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        # Dynamically determine the correct flattened size
        if self.flatten_size is None:
            self.flatten_size = x.view(x.size(0), -1).shape[1]
            self.fc1 = nn.Linear(self.flatten_size, 128) # Update fc1 layer based on input size
        
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) # No Sigmoid activation because we use BCEWithLogitsLoss
        #x = self.sigmoid(self.fc2(x))
        return x
    
# Initialize model, loss function, and optimizer
model = Simple3DCNN()
pos_weight = torch.tensor([1051/440]) # Weight positive class more heavily since (pcr: 0=1051 and 1=440)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
# def train_model(model, dataloader, criterion, optimizer, epochs=5):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for images, labels in dataloader:
#             labels = labels.view(-1, 1) # Reshape labels to match output
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")


# --- EVALUATION ---
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in dataloader:
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

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            labels = labels.view(-1, 1)
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
                labels = labels.view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/loss_curve.png")
    print("Loss curve saved as loss_curve.png")

# Train and evaluate the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=30)
evaluate_model(model, val_loader)
print("✅ Training complete!")