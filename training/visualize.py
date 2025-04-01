import argparse
import torch
from pathlib import Path
from models.mnist_model import MNISTModel
from utils.data_loader import get_mnist_data_loader
from utils.feature_space import visualize_features

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature space of trained models')
    parser.add_argument('--models', type=str, required=True, 
                        help='Comma-separated list of model paths')
    parser.add_argument('--method', type=str, default='tsne', 
                        choices=['tsne', 'pca', 'umap'],
                        help='Visualization method')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size for data loading')
    parser.add_argument('--num-samples', type=int, default=1000, 
                        help='Number of samples to visualize')
    return parser.parse_args()

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
    
    model_paths = [path.strip() for path in args.models.split(',')]
    for model_path in model_paths:

        model_name = Path(model_path).stem
        model = MNISTModel().to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            continue
        
        save_path = f"results/{model_name}_{args.method}_visualization.png"
        visualize_features(
            model=model,
            dataloader=subset_loader,
            device=device,
            method=args.method,
            loss_type=model_name,
            save_path=save_path
        )
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()