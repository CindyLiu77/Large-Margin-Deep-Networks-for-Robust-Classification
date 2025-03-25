import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.data_loader import get_cifar10_data_loader
from models.cifar_model import cifar_resnet_small, cifar_resnet_medium, cifar_resnet_large
from losses.margin_loss import SimpleLargeMarginLoss, LargeMarginLoss, MultiLayerMarginLoss
import argparse
import numpy as np
from pathlib import Path
from utils.feature_space import visualize_features
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training with Margin Loss')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--loss-type', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'simple_margin', 'margin', 'multi_layer_margin'],
                        help='Loss function to use')
    parser.add_argument('--gamma', type=float, default=10.0, help='Margin parameter')
    parser.add_argument('--norm', type=str, default='l2', choices=['l1', 'l2', 'linf'],
                        help='Norm type for margin computation')
    parser.add_argument('--aggregation', type=str, default='max', choices=['max', 'sum'],
                        help='Aggregation method for margin violations')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        choices=['sgd', 'adam', 'rmsprop'],
                        help='Optimizer to use')
    parser.add_argument('--model-size', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='Size of the ResNet model to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noisy-labels', type=float, default=0.0, 
                        help='Fraction of labels to corrupt (0.0 to 1.0)')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                        help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--layers', type=str, default='', 
                        help='Comma-separated list of layer indices for multi-layer margin')
    parser.add_argument('--visualize', action='store_true', 
                        help='Enable feature visualization')
    parser.add_argument('--vis-method', type=str, default='tsne', 
                        choices=['tsne', 'pca', 'umap'],
                        help='Visualization method for feature space')
    parser.add_argument('--vis-subset', type=int, default=1000,
                        help='Number of samples to use for visualization')
    return parser.parse_args()

def corrupt_labels(labels, corruption_fraction):
    """
    Corrupt a fraction of the labels randomly for noisy label training.
    """
    if corruption_fraction <= 0:
        return labels
    
    num_labels = len(labels)
    num_corrupt = int(corruption_fraction * num_labels)
    
    if num_corrupt <= 0:
        return labels
    
    corrupted_labels = labels.clone()
    
    # Randomly select indices to corrupt
    corrupt_indices = np.random.choice(num_labels, num_corrupt, replace=False)
    for idx in corrupt_indices:
        original_label = labels[idx].item()
        new_label = np.random.choice([l for l in range(10) if l != original_label])
        corrupted_labels[idx] = new_label
    
    return corrupted_labels

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            if hasattr(criterion, 'layers'):
                outputs, activations = model(images, return_activations=True)
                loss = criterion(outputs, labels, activations)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    train_loader, val_loader, test_loader = get_cifar10_data_loader(args.batch_size)
    
    # If using reduced data, create a subset of the training data
    if args.data_fraction < 1.0:
        train_data_list = []
        for batch in train_loader:
            images, labels = batch
            for i in range(len(images)):
                train_data_list.append((images[i], labels[i]))

        num_samples = int(len(train_data_list) * args.data_fraction)
        selected_indices = np.random.choice(len(train_data_list), num_samples, replace=False)
        selected_data = [train_data_list[i] for i in selected_indices]
        
        train_dataset = SimpleDataset(selected_data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Reduced training set size: {num_samples} samples")
    
    # Initialize model based on selected size
    if args.model_size == 'small':
        model = cifar_resnet_small().to(device)
    elif args.model_size == 'medium':
        model = cifar_resnet_medium().to(device)
    else:  # large
        model = cifar_resnet_large().to(device)
        
    print(f"Model size: {args.model_size}, parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Choose loss function
    if args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'simple_margin':
        criterion = SimpleLargeMarginLoss(gamma=args.gamma, aggregation=args.aggregation)
    elif args.loss_type == 'margin':
        criterion = LargeMarginLoss(gamma=args.gamma, norm=args.norm, 
                                  aggregation=args.aggregation)
    elif args.loss_type == 'multi_layer_margin':
        if not args.layers:
            # Use 5 evenly spaced layers by default as mentioned in the paper
            layers = [0, 2, 3, 4, 6]  # Input, 3 hidden layers, output
            print(f"Using default layers for multi-layer margin: {layers}")
        else:
            layers = [int(layer.strip()) for layer in args.layers.split(',')]
        criterion = MultiLayerMarginLoss(layers=layers, gamma=args.gamma, 
                                        norm=args.norm, aggregation=args.aggregation)
    
    # Choose optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                             momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, 
                                 weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    start_time = time.time()
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Apply noisy labels if specified
            if args.noisy_labels > 0:
                labels = corrupt_labels(labels, args.noisy_labels)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if args.loss_type == 'multi_layer_margin':
                outputs, activations = model(images, return_activations=True)
                loss = criterion(outputs, labels, activations)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = epoch_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Test set evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")
    
    # Create a descriptive model name for saving
    model_name = f"cifar10_model_{args.model_size}_{args.loss_type}"
    if args.noisy_labels > 0:
        model_name += f"_noisy{int(args.noisy_labels*100)}pct"
    if args.data_fraction < 1.0:
        model_name += f"_data{int(args.data_fraction*100)}pct"
    
    # Save model
    torch.save(model.state_dict(), f"checkpoints/{model_name}.pth")
    
    # Plot results
    plot_test_results(model, test_loader, device, model_name)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name)

    # Feature visualization if enabled
    if args.visualize:
        print(f"Visualizing features using {args.vis_method}...")
        
        # Create a subset of test data for visualization
        subset_indices = np.random.choice(len(test_loader.dataset), args.vis_subset, replace=False)
        subset = torch.utils.data.Subset(test_loader.dataset, subset_indices)
        subset_loader = torch.utils.data.DataLoader(
            subset, 
            batch_size=args.batch_size,
            shuffle=False
        )
        
        vis_save_path = f"results/{model_name}_{args.vis_method}_visualization.png"
        visualize_features(
            model=model,
            dataloader=subset_loader, 
            device=device,
            method=args.vis_method,
            loss_type=args.loss_type,
            save_path=vis_save_path
        )
        print(f"Feature visualization saved to {vis_save_path}")

def plot_test_results(model, test_loader, device, model_name, num_images=10):
    """
    Visualize model predictions on test data.
    """
    model.eval()
    
    # Get a batch of images and predictions
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images[:num_images].to(device), labels[:num_images].to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Plot images with predictions
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        # Denormalize the image
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        pred_label = classes[predictions[i].item()]
        true_label = classes[labels[i].item()]
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_test_results.png")
    plt.close()

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name):
    """
    Plot training and validation curves.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_training_curves.png")
    plt.close()

if __name__ == "__main__":
    main()