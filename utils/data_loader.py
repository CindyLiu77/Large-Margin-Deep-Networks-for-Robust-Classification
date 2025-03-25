import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data
from torch.utils.data import DataLoader, random_split

def get_mnist_data_loader(batch_size, val_split=0.1, num_workers=4, pin_memory=True):
    """
    Load MNIST dataset and return train, validation and test dataloaders.
    
    Args:
        batch_size: Number of samples in a batch
        val_split: Percentage of training data to use for validation
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU data transfer
        
    Returns:
        train_loader: Dataloader for training data
        val_loader: Dataloader for validation data
        test_loader: Dataloader for test data
    """
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset = datasets.MNIST(
        root="./data", 
        train=True, 
        transform=transform, 
        download=True
    )
    
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = data.random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=generator
    )

    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_dataset = datasets.MNIST(
        root="./data", 
        train=False, 
        transform=transform, 
        download=True
    )
    
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

# def get_cifar10_data_loader(batch_size, val_split=0.1, num_workers=4, pin_memory=True):
#     """
#     Load CIFAR-10 dataset and return train, validation and test dataloaders.
    
#     Args:
#         batch_size: Number of samples in a batch
#         val_split: Percentage of training data to use for validation
#         num_workers: Number of worker processes for data loading
#         pin_memory: Pin memory for faster GPU data transfer
        
#     Returns:
#         train_loader: Dataloader for training data
#         val_loader: Dataloader for validation data
#         test_loader: Dataloader for test data
#     """
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     # Load full CIFAR-10 dataset
#     full_train_dataset = datasets.CIFAR10(
#         root="./data", 
#         train=True, 
#         transform=transform_train, 
#         download=True
#     )
    
#     # Split into training and validation
#     train_size = int((1 - val_split) * len(full_train_dataset))
#     val_size = len(full_train_dataset) - train_size
    
#     # Use a fixed generator for reproducibility
#     generator = torch.Generator().manual_seed(42)
#     train_dataset, val_dataset = random_split(
#         full_train_dataset, 
#         [train_size, val_size],
#         generator=generator
#     )

#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )
    
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )
    
#     test_dataset = datasets.CIFAR10(
#         root="./data", 
#         train=False, 
#         transform=transform_test, 
#         download=True
#     )
    
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )

#     return train_loader, val_loader, test_loader