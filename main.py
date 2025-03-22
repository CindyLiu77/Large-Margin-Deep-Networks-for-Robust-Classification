import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset_loader import DatasetLoader

np.random.seed(42)
tf.random.set_seed(42)

def test_dataset_loader():
    print("="*80)
    print("TESTING DATASET LOADER")
    print("="*80)

    print("\nTesting MNIST dataset loading...")
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        print("Data directory: ", data_dir)

        mnist_loader = DatasetLoader(dataset_name='mnist', data_dir=data_dir)
        print(f"\nMNIST dataset loaded successfully!")
        print(f"Training set shape: {mnist_loader.x_train.shape}")
        print(f"Validation set shape: {mnist_loader.x_val.shape}")
        print(f"Test set shape: {mnist_loader.x_test.shape}")

        # Test data subset creation
        subset_percentage = 0.1
        print(f"\nCreating {subset_percentage*100}% subset of training data...")
        x_subset, y_subset, y_subset_onehot = mnist_loader.prepare_subset(subset_percentage)
        print(f"Subset shapes - X: {x_subset.shape}, y: {y_subset.shape}, y_onehot: {y_subset_onehot.shape}")
        
        # Test noisy label creation
        noise_percentage = 0.2
        print(f"\nCreating labels with {noise_percentage*100}% noise...")
        x_noisy, y_noisy, y_noisy_onehot, y_original = mnist_loader.add_label_noise(noise_percentage)
        num_changed = np.sum(y_noisy != y_original)
        print(f"Noisy labels created. Changed {num_changed} out of {len(y_original)} labels ({num_changed/len(y_original)*100:.2f}%)")
        
        # # Test data generator
        # batch_size = 32
        # print(f"\nCreating data generators with batch size {batch_size}...")
        # # Regular generator
        # train_dataset = mnist_loader.create_data_generator(
        #     mnist_loader.x_train, 
        #     mnist_loader.y_train_onehot, 
        #     batch_size=batch_size
        # )
        
        # # Augmented generator
        # augmented_dataset = mnist_loader.create_data_generator(
        #     mnist_loader.x_train, 
        #     mnist_loader.y_train_onehot, 
        #     batch_size=batch_size,
        #     augment=True
        # )
        
        # # Test visualization functions
        # print("\nVisualizing sample images...")
        # mnist_loader.visualize_samples(num_samples=8)
        
        # # Test visualization of noisy labels
        # print("\nVisualizing original vs noisy labels...")
        # mnist_loader.compare_original_and_noisy(y_original, y_noisy, num_samples=8)
        
        print("\nMNIST dataset tests completed successfully!")
        
    except Exception as e:
        print(f"Error testing MNIST dataset: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("Testing CIFAR-10 dataset loading...")
    try:
         # Initialize dataset loader
        cifar_loader = DatasetLoader(dataset_name='cifar10', data_dir=data_dir)
        
        # Print dataset information
        print(f"\nCIFAR-10 dataset loaded successfully!")
        print(f"Training set shape: {cifar_loader.x_train.shape}")
        print(f"Validation set shape: {cifar_loader.x_val.shape}")
        print(f"Test set shape: {cifar_loader.x_test.shape}")
        
        # Test visualizing samples
        # print("\nVisualizing CIFAR-10 sample images...")
        # cifar_loader.visualize_samples(num_samples=8)
        
        print("\nCIFAR-10 dataset tests completed successfully!")
        
    except Exception as e:
        print(f"Error testing CIFAR-10 dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL DATASET LOADER TESTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    start_time = time.time()
    directories = [
        'data',
        'losses',
        'results',
        'results/figures',
        'results/models',
        'results/logs',
        'utils'
    ]
    
    for directory in directories:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    
    # Run dataset loader tests
    test_dataset_loader()
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")