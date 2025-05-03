import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_class_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_class_weight = pos_class_weight  # Additional weighting for positive class
        
    def forward(self, inputs, targets):
        # Create weight tensor
        weights = torch.ones_like(targets)
        weights[targets == 1] = self.pos_class_weight  # Higher weight for positive class
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Apply class weights
        weighted_focal_loss = weights * focal_loss
        
        return weighted_focal_loss.mean()