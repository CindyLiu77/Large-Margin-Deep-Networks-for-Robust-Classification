import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import PCA
import torch
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, device, title="Decision Boundary", save_path="results/decision_boundary.png"):
    
    """
    Plot the decision boundary of a model on a 2D representation of the data.
    
    Args:
        model: PyTorch model with a forward method that outputs logits/predictions
        X: Input features (numpy array or torch tensor)
        y: True labels (numpy array or torch tensor)
        device: torch device (cuda or cpu)
        title: Title for the plot
        filename: If provided, save the plot to this file
    """

    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.long)

    model.eval()
    model.to(device)
    X, y = X.to(device), y.to(device)
