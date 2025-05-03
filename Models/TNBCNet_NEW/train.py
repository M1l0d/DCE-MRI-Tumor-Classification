import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm
from evaluate import validate
from utils import set_seed
from focal_loss import FocalLoss  # Import the new FocalLoss class

def calculate_pos_weight(train_loader):
    """Calculate positive class weight based on class distribution"""
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    
    all_labels = torch.cat(all_labels).numpy()
    n_samples = len(all_labels)
    n_positives = np.sum(all_labels)
    
    if n_positives == 0:
        return torch.tensor([1.0])  # Default if no positives
    
    # Weight = number of negatives / number of positives
    pos_weight = (n_samples - n_positives) / n_positives
    print(f"Class balance - Positive weight: {pos_weight:.2f} (Neg:Pos ratio = {pos_weight:.2f}:1)")
    return torch.tensor([pos_weight])

def get_scheduler(config, optimizer, steps_per_epoch=None):
    """Get appropriate learning rate scheduler based on config"""
    if config.get('use_cosine_scheduler', False):
        # CosineAnnealingWarmRestarts for cycling learning rate
        return CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config.get('cosine_T_0', 10), 
            T_mult=config.get('cosine_T_mult', 2),
            eta_min=1e-6
        )
    elif config.get('use_one_cycle', False) and steps_per_epoch:
        # OneCycleLR for 1cycle policy with warmup
        return OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            steps_per_epoch=steps_per_epoch,
            epochs=config['epochs'],
            pct_start=0.3  # Spend 30% of time warming up
        )
    else:
        # Default: ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

def train_one_fold(train_loader, val_loader, device, config):
    set_seed(config['seed'])

    # Import from appropriate module based on config
    if config.get('use_resnet', True):
        from model_resnet import TNBCNet
    else:
        from model import TNBCNet

    model = TNBCNet(
        in_channels=config['in_channels'],
        backbone=config.get('backbone', 'resnet18'),
        dropout_rate=config.get('dropout_rate', 0.5)
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)  # Add weight decay for regularization
    )
    
    # Get appropriate scheduler
    scheduler = get_scheduler(
        config, 
        optimizer, 
        steps_per_epoch=len(train_loader)
    )
    
    # Calculate and apply class weights
    pos_weight = calculate_pos_weight(train_loader)
    
    # Choose loss function - Focal Loss or BCE
    if config.get('use_focal_loss', True):
        criterion = FocalLoss(
        alpha=config.get('focal_alpha', 0.75),
        gamma=config.get('focal_gamma', 2.0)
        # Remove the pos_weight parameter
        )
        print(f"Using Focal Loss with alpha={config.get('focal_alpha', 0.75)}, gamma={config.get('focal_gamma', 2.0)}")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        print("Using BCEWithLogitsLoss")

    # Table header - OUTSIDE the epoch loop
    print("\n| Epoch | Train Loss | Val Loss | Val AUC | Val Accuracy | Val Precision | Val Recall | Val F1 |")
    print("|-------|------------|----------|---------|--------------|---------------|------------|--------|")

    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['epochs']}]", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            
            # Update LR for OneCycleLR scheduler which needs step per batch
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        val_loss, val_metrics = validate(model, val_loader, device, criterion)
        
        # Step scheduler (except OneCycleLR which is updated every batch)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(epoch + 1)  # +1 because epoch starts at 0

        # Print table row - simple format for each epoch
        print(f"| {epoch+1:5d} | {avg_train_loss:.6f} | {val_loss:.6f} | {val_metrics['auc']:.6f} | {val_metrics['accuracy']:.6f} | {val_metrics['precision']:.6f} | {val_metrics['recall']:.6f} | {val_metrics['f1']:.6f} |")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    return best_model_state