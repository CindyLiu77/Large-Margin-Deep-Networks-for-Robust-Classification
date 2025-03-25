import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sklearn.decomposition
from models.mnist_model import MNISTModel
from utils.data_loader import get_mnist_data_loader
import umap

def extract_features(model, dataloader, device):
    
    '''
    Extract features from the model for the given dataloader
    Args:
        model: PyTorch model
        dataloader: PyTorch dataloader
        device: torch device
        Returns:
        features: Extracted features
        labels: True labels
    '''

    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, _labels in dataloader:
            images, _labels = images.to(device), _labels.to(device)
            if hasattr(model, 'return_activations'):
                _features, _ = model(images, return_activations=True)
            else:
                _features = model(images)
            features.append(_features)
            labels.append(_labels)
    return torch.cat(features).cpu().numpy(), torch.cat(labels).cpu().numpy()

def plot_tsne(features, labels, num_classes, method='tsne', loss_type='cross_entropy', save_path=None):

    ''' 
    Plot t-SNE visualization of the features
    Args:
        features: Extracted features
        labels: True labels
        num_classes: Number of classes
        method: Dimensionality reduction method ('tsne' or 'pca')
        loss_type: Loss type ('cross_entropy' or 'margin')
    '''

    print(f"Features shape before dimensionality reduction: {features.shape}")
    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        features = tsne.fit_transform(features)
    else: # I dont think we need this so ignore
        pca = sklearn.decomposition.PCA(n_components=2)
        features = pca.fit_transform(features)
    
    print(f"Features shape after dimensionality reduction: {features.shape}")

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        print(f"Plotting class {i}...")
        plt.scatter(features[labels==i, 0], features[labels==i, 1], label=str(i))
    plt.legend()
    plt.title(f'Feature Visualization using {method.upper()} for {loss_type}')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_umap(features, labels, num_classes, loss_type='cross_entropy', save_path=None):

    '''''
    Plot UMAP visualization of the features'
    Args:
        features: Extracted features
        labels: True labels
        num_classes: Number of classes
        loss_type: Loss type ('cross_entropy' or 'margin')
    '''

    print(f"Features shape before dimensionality reduction: {features.shape}")
    umap_ = umap.UMAP(n_components=2)
    features = umap_.fit_transform(features)
    print(f"Features shape after dimensionality reduction: {features.shape}")

    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        print(f"Plotting class {i}...")
        plt.scatter(features[labels==i, 0], features[labels==i, 1], label=str(i))
    plt.legend()
    plt.title(f'Feature Visualization using UMAP for {loss_type}')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_features(model, dataloader, device, method='tsne', loss_type='cross_entropy', save_path=None, num_classes=10):
    """
    Extract features and visualize them
    Args:
        model: PyTorch model
        dataloader: PyTorch dataloader
        device: torch device
        method: Visualization method ('tsne', 'pca', 'umap')
        loss_type: Loss type name for title
        save_path: Path to save the figure
    """
    features, labels = extract_features(model, dataloader, device)
    
    if method == 'umap':
        plot_umap(features, labels, num_classes=num_classes, loss_type=loss_type, save_path=save_path)
    else:
        plot_tsne(features, labels, num_classes=num_classes, method=method, loss_type=loss_type, save_path=save_path)

# Legacy Test

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ce_model = MNISTModel().to(device)
# ce_model.load_state_dict(torch.load("checkpoints/mnist_model_cross_entropy_loss.pth"))

# margin_model = MNISTModel().to(device)
# margin_model.load_state_dict(torch.load("checkpoints/mnist_model_margin_loss.pth"))

# print("Loading MNIST test dataset...")

# test_loader, _, _ = get_mnist_data_loader(batch_size=128)

# print("Extracting features for CE model...")

# ce_features, ce_labels = extract_features(ce_model, test_loader, device)
# margin_features, margin_labels = extract_features(margin_model, test_loader, device)

# print("Plotting t-SNE visualization...")

# plot_tsne(ce_features, ce_labels, num_classes=10, method='tsne', loss_type='cross_entropy')
# plot_tsne(margin_features, margin_labels, num_classes=10, method='tsne', loss_type='margin')

# print("PLotting UMAP visualization...")

# plot_umap(ce_features, ce_labels, num_classes=10, loss_type='cross_entropy')
# plot_umap(margin_features, margin_labels, num_classes=10, loss_type='margin')

