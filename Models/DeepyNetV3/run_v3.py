# run_v2.py

import os
import time
import random
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch import nn, optim

from dataset import MRIDataset, load_clinical_metadata
from deepynet_v3 import DeepyNetV3
from train import train_model
from evaluate import evaluate_model

# ---------------------
# 🔧 Argument Parser
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_mode", type=str, default="delta", choices=["t2", "delta", "delta2"])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ---------------------
# 🔒 Seeding
# ---------------------
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------
# 📂 Configs
# ---------------------
VAL_SPLIT = 0.2
NUM_WORKERS = 4
PATIENCE = 10
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔥 Device: {DEVICE} | Mode: {args.input_mode} | Seed: {args.seed}")

# ---------------------
# 📂 Paths
# ---------------------
TIMEPOINT_DIRS = [
    "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0000",
    "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0001",
    "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0002"
]
LABELS_PATH = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/filtered_clinical_and_imaging_info_pcr.xlsx"

# ---------------------
# 📁 Logging dir
# ---------------------
timestamp = time.strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("DeepyNetV2_1_results", f"seed_{args.seed}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)
best_model_path = os.path.join(run_dir, "best_model.pt")

# ---------------------
# 📊 Load clinical
# ---------------------
labels_df, labels_dict, clinical_dict, clinical_cols = load_clinical_metadata(LABELS_PATH)

dataset = MRIDataset(
    timepoint_dirs=TIMEPOINT_DIRS,
    labels_df=labels_df,
    input_mode=args.input_mode,
    augment=True,
    clinical_cols=clinical_cols
)

# ---------------------
# ✂️ Train/Val Split
# ---------------------
val_size = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.augment = True

# ---------------------
# ⚖️ Weighted Sampling
# ---------------------
train_labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
class_counts = torch.tensor([train_labels.count(0), train_labels.count(1)], dtype=torch.float32)
weights = 1. / class_counts
sample_weights = [weights[int(lbl)] for lbl in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)

# ---------------------
# 🧠 Model & Optimizer
# ---------------------
in_channels = 3 if args.input_mode == "delta2" else 2 if args.input_mode == "delta" else 1
tabular_dim = dataset.num_clinical_features
model = DeepyNetV3(in_channels=in_channels, tabular_dim=tabular_dim, dropout=DROPOUT).to(DEVICE)

pos_weight = class_counts[0] / class_counts[1]
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

# ---------------------
# 🚀 Train
# ---------------------
print("🚀 Training DeepyNetV3 ...")
train_model(
    model=model,
    loaders=(train_loader, val_loader),
    optimizer=optimizer,
    criterion=criterion,
    device=DEVICE,
    max_epochs=args.epochs,
    patience=PATIENCE,
    save_path=best_model_path,
    run_dir=run_dir
)

# ---------------------
# 🔍 Evaluate
# ---------------------
print("🔍 Evaluating...")
model.load_state_dict(torch.load(best_model_path))
evaluate_model(model, val_loader, device=DEVICE, save_path=os.path.join(run_dir, "metrics.json"))
print(f"✅ Run complete. Results saved to: {run_dir}")
