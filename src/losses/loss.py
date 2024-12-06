# Path: src/losses/loss.py
import torch.nn as nn

def get_loss_function():
    """
    Returns the Binary Cross-Entropy Loss function.
    """
    return nn.BCEWithLogitsLoss()
