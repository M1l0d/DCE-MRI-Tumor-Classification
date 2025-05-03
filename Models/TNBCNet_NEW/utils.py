import random
import numpy as np
import torch

def set_seed(seed):
    """Set random seed for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_metrics(labels, preds):
    """Calculate standard classification metrics given true labels and predicted probabilities."""
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    # Convert predicted probabilities to binary predictions
    preds_binary = (preds > 0.5).astype(np.float32)

    metrics = {
        'auc': roc_auc_score(labels, preds),
        'accuracy': accuracy_score(labels, preds_binary),
        'precision': precision_score(labels, preds_binary, zero_division=0),
        'recall': recall_score(labels, preds_binary, zero_division=0),
        'f1': f1_score(labels, preds_binary, zero_division=0)
    }
    return metrics
