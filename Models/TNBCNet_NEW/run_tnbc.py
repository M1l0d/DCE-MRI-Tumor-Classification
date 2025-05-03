import argparse
import torch
import os
import numpy as np
from dataset_tnbc import get_cross_validation_loaders
from model_resnet import TNBCNet  # Updated import to use ResNet model
from train import train_one_fold
from evaluate import validate, evaluate_model
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="TNBC Classification Pipeline")
    parser.add_argument("--data_dir", type=str, default="/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia", help="Path to the MAMA-MIA dataset directory")
    parser.add_argument("--input_mode", type=str, default="delta2", choices=["t2", "delta", "delta2", "t0t2"], help="Input mode for the model")
    parser.add_argument("--mri_mode", type=str, default="fixed_crop_128_padded_matched", help="MRI preprocessing mode (e.g., fixed_crop_128_padded_matched)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--use_segmentation", action="store_true", help="Use segmentation masks as input channel")
    parser.add_argument("--use_pe_map", action="store_true", help="Use PE maps as input channel")
    parser.add_argument("--use_ser_map", action="store_true", help="Use SER maps as input channel")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results and checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"], help="Backbone architecture")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for regularization")
    parser.add_argument("--use_augmentation", action="store_true", help="Use data augmentation")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use Focal Loss instead of BCE")
    parser.add_argument("--focal_alpha", type=float, default=0.75, help="Alpha weighting in Focal Loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma focusing parameter in Focal Loss")
    parser.add_argument("--use_ensemble", action="store_true", help="Use model ensemble from all CV folds")
    parser.add_argument("--classification_threshold", type=float, default=0.35, help="Threshold for binary classification")
    
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, choices=["cuda", "cpu"], help="Device to run the model on (cuda or cpu)")

    return parser.parse_args()

def get_predictions(model, loader, device):
    """Get predictions from a model for a given data loader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_labels)

def ensemble_evaluate(all_fold_preds, all_fold_labels, threshold=0.5):
    """Evaluate ensemble predictions"""
    # Average predictions from all folds
    ensemble_preds = np.mean(all_fold_preds, axis=0)
    ensemble_labels = all_fold_labels[0]  # All labels should be the same
    
    # Binarize predictions
    binary_preds = (ensemble_preds > threshold).astype(np.float32)
    
    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(ensemble_labels, ensemble_preds),
        'accuracy': accuracy_score(ensemble_labels, binary_preds),
        'precision': precision_score(ensemble_labels, binary_preds, zero_division=0),
        'recall': recall_score(ensemble_labels, binary_preds, zero_division=0),
        'f1': f1_score(ensemble_labels, binary_preds, zero_division=0)
    }
    
    return metrics

def main():
    args = parse_args()

    # Force CPU if CUDA is requested but not available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ Warning: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare for ensemble evaluation
    fold_models = []
    fold_val_loaders = []
    fold_predictions = []
    fold_labels = []

    # Get all cross-validation loaders at once to keep consistent data splits
    cv_data = list(get_cross_validation_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_mode=args.input_mode,
        mri_mode=args.mri_mode,
        label_column="tnbc_label",
        use_segmentation=args.use_segmentation,
        use_pe_map=args.use_pe_map,
        use_ser_map=args.use_ser_map,
        n_splits=args.n_splits,
        use_augmentation=args.use_augmentation
    ))

    for fold, (train_loader, val_loader) in enumerate(cv_data):
        print(f"\n🚀 Starting training for fold {fold + 1}/{args.n_splits}")

        model = TNBCNet(
            in_channels=train_loader.dataset[0][0].shape[0],
            backbone=args.backbone,
            dropout_rate=args.dropout_rate
        ).to(args.device)

        # Create a config dictionary
        config = {
            'in_channels': train_loader.dataset[0][0].shape[0],
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'seed': args.seed,
            'use_cosine_scheduler': True,
            'warmup_epochs': 5,
            'cosine_T_0': 10,
            'cosine_T_mult': 2,
            'backbone': args.backbone,
            'dropout_rate': args.dropout_rate,
            'weight_decay': 1e-4,
            'use_focal_loss': args.use_focal_loss,
            'focal_alpha': args.focal_alpha,
            'focal_gamma': args.focal_gamma
        }

        # Train the model
        best_model_state = train_one_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            config=config
        )

        # Load the best model state
        model.load_state_dict(best_model_state)
        
        # Save the model to disk
        torch.save(
            best_model_state, 
            os.path.join(args.output_dir, f"tnbc_model_fold_{fold+1}.pth")
        )

        # Individual fold evaluation
        print(f"\n✅ Evaluation for fold {fold + 1}:")
        evaluate_model(model, val_loader, device=args.device, threshold=args.classification_threshold)
        
        # Store for ensemble evaluation
        fold_models.append(model)
        fold_val_loaders.append(val_loader)
        
        # Get predictions for this fold
        fold_preds, fold_lbls = get_predictions(model, val_loader, args.device)
        fold_predictions.append(fold_preds)
        fold_labels.append(fold_lbls)

    # Ensemble evaluation if enabled and we have multiple folds
    if args.use_ensemble and len(fold_models) > 1:
        print("\n🌟 Ensemble Evaluation (Average of all folds):")
        ensemble_metrics = ensemble_evaluate(fold_predictions, fold_labels, threshold=args.classification_threshold)
        
        print("\n| Metric    | Value     |")
        print("|-----------|-----------|")
        for metric, value in ensemble_metrics.items():
            print(f"| {metric.capitalize():9s} | {value:.6f} |")

        # Save ensemble predictions
        np.save(os.path.join(args.output_dir, "ensemble_predictions.npy"), np.array(fold_predictions))
        np.save(os.path.join(args.output_dir, "ensemble_labels.npy"), np.array(fold_labels))

if __name__ == "__main__":
    main()