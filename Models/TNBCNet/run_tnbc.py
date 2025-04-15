import argparse
import os
import torch
import torch.nn as nn
from datetime import datetime

# from model import TNBCNet
from model_resnet import TNBCNet, SimpleConv3DNet
from dataset_tnbc import get_dataloaders, visualize_sample, get_cross_validation_loaders
from train import train_model, find_lr
from train import SmoothedBCEWithLogitsLoss, FocalLoss
from evaluate import evaluate_model, visualize_features
from utils import set_seed, count_parameters

# --- Initialize Weights ---
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--input_mode", choices=["image"], default="image")
    parser.add_argument("--mri_mode", choices=["t2", "delta", "delta2", "t0t2"], default="delta")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia")
    parser.add_argument("--label_column", type=str, default="tnbc")
    parser.add_argument("--use_segmentation_channel", action="store_true")
    parser.add_argument("--use_pe_map", action="store_true", help="Use PE map as extra channel")
    parser.add_argument("--debug", action="store_true", help="Run a single fold for debugging")
    parser.add_argument("--use_simple_model", action="store_true", help="Use a simpler model instead of TNBCNet")
    args = parser.parse_args()

    # --- Set Seed and Device ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device} | Seed: {args.seed}")

    # --- Cross-Validation Loaders ---
    print("🚀 Generating cross-validation loaders...")
    cv_loaders = get_cross_validation_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_mode=args.input_mode,
        mri_mode=args.mri_mode,
        label_column=args.label_column,
        use_segmentation=args.use_segmentation_channel,
        use_pe_map=args.use_pe_map,
        n_splits=5
    )

    # --- Model Initialization ---
    # delta mode = t2 + delta1 → 2 base channels
    if args.mri_mode == "delta":
        num_base_channels = 2
    elif args.mri_mode == "delta2":
        num_base_channels = 3
    elif args.mri_mode in ["t0t2", "t2"]:
        num_base_channels = 2 if args.mri_mode == "t0t2" else 1
    else:
        raise ValueError("Unsupported mri_mode")

    in_channels = num_base_channels + int(args.use_segmentation_channel) + int(args.use_pe_map)
    
    print(f"📦 Initializing model with {in_channels} input channels...")
    model = TNBCNet(in_channels=in_channels).to(device)

    # --- Debugging: Visualize Data ---
    print(f"🔬 Visualizing a sample from the training set:")
    for fold, (train_loader, _) in enumerate(cv_loaders):
        visualize_sample(train_loader.dataset, 0)  # Visualize the first sample
        visualize_sample(train_loader.dataset, 1)  # Visualize the second sample
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        break  # Visualize only once (from the first fold)

    # --- Cross-Validation Training ---
    for fold, (train_loader, val_loader) in enumerate(cv_loaders):
        
        # --- Initialize Model ---
        if args.use_simple_model:
            print("Using simple model")
            model = SimpleConv3DNet(in_channels=in_channels).to(device)
        else:
            print("Using TNBCNet model")
            model = TNBCNet(in_channels=in_channels, backbone="resnet18").to(device)

        if args.debug:
            print("🐞 Debug mode: Running only one fold")
            # Get the first fold only
            for fold, (train_loader, val_loader) in enumerate(cv_loaders):
                break
        else:
            print(f"🔄 Fold {fold + 1}")

            # --- Initialize Custom Weights ---
            model.apply(init_weights)
            print("✅ Applied custom weight initialization")

            # --- Initialize Optimizer, and Criterion ---

            labels = [y.item() for _, y in train_loader.dataset]
            pos_count = sum(1 for l in labels if l > 0.5)
            neg_count = len(labels) - pos_count

            print(f"Class distribution: {pos_count} positives, {neg_count} negatives")

            # Calculate pos_weight for BCE loss
            pos_weight = torch.tensor([neg_count / max(1, pos_count)]).to(device)
            print(f"Using pos_weight: {pos_weight.item():.2f}")

            # Use weighted BCE loss instead of Focal Loss
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)

            # --- Learning Rate Finder ---
            print("🔍 Finding optimal learning rate...")
            find_lr(model, train_loader, optimizer, criterion, device, init_lr=1e-10, final_lr=1e-6, beta=0.98)

            # --- Training ---
            run_dir = f"TNBC_results/fold_{fold + 1}_seed_{args.seed}_{timestamp}"
            os.makedirs(run_dir, exist_ok=True)
            model_path = os.path.join(run_dir, "best_model.pt")

            print("🚀 Training...")
            train_model(model, (train_loader, val_loader), optimizer, criterion, device, args.epochs, model_path, run_dir)

            # --- Evaluation ---
            print("🔍 Evaluating...")
            evaluate_model(model, val_loader, model_path, device, run_dir)

            # --- Feature Visualization ---
            print("🔍 Visualizing features...")
            visualize_features(model, val_loader, device, run_dir)

    # --- Debugging: Overfitting Test ---
    print("🔍 Training with overfitting debug...")
    small_train_dataset = torch.utils.data.Subset(train_loader.dataset, list(range(10)))
    small_train_loader = torch.utils.data.DataLoader(small_train_dataset, batch_size=args.batch_size, shuffle=True)

    # Calculate class distribution for small dataset
    small_labels = [train_loader.dataset[i][1].item() for i in range(10)]
    small_pos_count = sum(1 for l in small_labels if l > 0.5)
    small_neg_count = len(small_labels) - small_pos_count

    # Calculate pos_weight for BCE loss
    small_pos_weight = torch.tensor([small_neg_count / max(1, small_pos_count)]).to(device)
    print(f"Small dataset class distribution: {small_pos_count} positives, {small_neg_count} negatives")
    print(f"Using small dataset pos_weight: {small_pos_weight.item():.2f}")

    # Define criterion for overfitting test
    criterion = nn.BCEWithLogitsLoss(pos_weight=small_pos_weight)

    model = SimpleConv3DNet(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    overfit_dir = f"TNBC_results/overfit_seed_{args.seed}_{timestamp}"
    os.makedirs(overfit_dir, exist_ok=True)
    overfit_model_path = os.path.join(overfit_dir, "overfit_model.pt")

    train_model(model, (small_train_loader, val_loader), optimizer, criterion, device, args.epochs, overfit_model_path, overfit_dir)