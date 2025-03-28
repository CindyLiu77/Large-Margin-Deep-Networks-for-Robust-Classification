import argparse
import torch
import glob
from pathlib import Path
from models.mnist_model import MNISTModel
from utils.data_loader import get_mnist_data_loader
from utils.feature_space import visualize_features, extract_features
import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn.manifold import TSNE
import umap

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature space of trained models')
    parser.add_argument('--loss-type', type=str, required=True, 
                        help='Comma-separated list of loss types: e.g. "cross_entropy,multi_layer_margin"')
    parser.add_argument('--method', type=str, default='tsne', 
                        choices=['tsne', 'pca', 'umap'],
                        help='Visualization method')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size for data loading')
    parser.add_argument('--num-samples', type=int, default=1000, 
                        help='Number of samples to visualize')
    return parser.parse_args()

def find_checkpoint_for_loss(loss_type):
    """
    Find the most relevant checkpoint for a given loss type.
    Example: "multi_layer_margin" might match "mnist_model_multi_layer_margin_noisy30%.pth"
    """
    pattern = f"checkpoints/mnist_model_{loss_type}*.pth"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for loss type '{loss_type}' using pattern: {pattern}")
    return matches[-1]  # return latest match

def main():
    args = parse_args()

    Path("results").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    _, _, test_loader = get_mnist_data_loader(batch_size=args.batch_size)
    
    subset_indices = torch.randperm(len(test_loader.dataset))[:args.num_samples]
    subset = torch.utils.data.Subset(test_loader.dataset, subset_indices)
    subset_loader = torch.utils.data.DataLoader(
        subset, 
        batch_size=args.batch_size,
        shuffle=False
    )

    loss_types = [lt.strip() for lt in args.loss_type.split(',')]
    features_list = []
    labels_list = []

    for loss_type in loss_types:
        try:
            model_path = find_checkpoint_for_loss(loss_type)
            print(f"Using checkpoint: {model_path}")
        except FileNotFoundError as e:
            print(e)
            continue

        model = MNISTModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        features, labels = extract_features(model, subset_loader, device)
        features_list.append(features)
        labels_list.append(labels)

    fig, axs = plt.subplots(1, len(loss_types), figsize=(6 * len(loss_types), 5))
    if len(loss_types) == 1:
        axs = [axs]  # ensure it's iterable

    for i, (features, labels, loss_type) in enumerate(zip(features_list, labels_list, loss_types)):
        if args.method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif args.method == 'pca':
            reducer = sklearn.decomposition.PCA(n_components=2)
        elif args.method == 'umap':
            reducer = umap.UMAP(n_components=2)

        print(f"Reducing features for {loss_type} using {args.method}")
        reduced = reducer.fit_transform(features)

        for cls in range(10):
            axs[i].scatter(reduced[labels==cls, 0], reduced[labels==cls, 1], label=str(cls), alpha=0.6, s=10)
        axs[i].set_title(f"{loss_type} ({args.method.upper()})")
        axs[i].legend()

    plt.tight_layout()
    save_path = f"results/side_by_side_{args.method}_visualization.png"
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
