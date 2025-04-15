# run.py
import os
import time
import torch
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_sample_weight
from deepynet import DeepyNet3D
from dataset import MRIDataset
from train import train_model
from evaluate import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepyNet3D on DCE-MRI breast cancer dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--input_mode", type=str, default="delta", choices=["t2_only", "delta", "full"])
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos_weight", type=float, default=2.0)
    parser.add_argument("--save_dir", type=str, default="runs/")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device} | Mode: {args.input_mode}")

    # --- Paths ---
    TIMEPOINT_DIRS = [
        "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0000",
        "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0001",
        "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0002"
    ]
    LABELS_CSV = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/filtered_clinical_and_imaging_info_pcr.xlsx"

    df = pd.read_excel(LABELS_CSV).dropna(subset=["pcr"])
    labels_dict = dict(zip(df["patient_id"], df["pcr"]))
    valid_ids = set(df["patient_id"])

    dataset = MRIDataset(TIMEPOINT_DIRS, labels_dict, valid_ids, input_mode=args.input_mode, augment=False)
    val_len = int(args.val_split * len(dataset))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_set.dataset.augment = True

    train_labels = [train_set[i][1].item() for i in range(len(train_set))]
    weights = compute_sample_weight(class_weight='balanced', y=train_labels)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Model ---
    in_channels = {
        "t2_only": 1,
        "delta": 2,
        "full": 3
    }[args.input_mode]

    model = DeepyNet3D(in_channels=in_channels).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Run Directory ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.save_dir, f"{args.input_mode}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, "best_model.pt")

    # --- Training ---
    print("🚀 Training DeepyNet3D...")
    train_model(
        model=model,
        loaders=(train_loader, val_loader),
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_epochs=args.epochs,
        patience=args.patience,
        save_path=best_model_path
    )

    # --- Evaluation ---
    print("🔍 Evaluating...")
    model.load_state_dict(torch.load(best_model_path))
    evaluate_model(model, val_loader, device, save_path=os.path.join(run_dir, "metrics.json"))

    print(f"✅ Run complete. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
