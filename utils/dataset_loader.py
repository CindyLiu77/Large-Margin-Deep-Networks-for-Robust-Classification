import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import os
import sys

np.random.seed(42)
tf.random.set_seed(42)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


print("Starting dataset loader script...")

class DatasetLoader:
    def __init__(self, dataset_name='mnist', data_dir=None):
        """
        Initialize the dataset loader
        
        Args:
            dataset_name (str): Name of the dataset - 'mnist', 'cifar10', 'imagenet'?
            data_dir (str): Directory to store the dataset
        """
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in ['mnist', 'cifar10', 'imagenet']:
            raise ValueError("Invalid dataset name. Choose from 'mnist', 'cifar10', 'imagenet'")
        
        # set data dir
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        else:
            self.data_dir = data_dir

        os.makedirs(self.data_dir, exist_ok=True)
        os.environ['KERAS_HOME'] = self.data_dir # set tensorflow environment variable

        print(f"Loading {self.dataset_name} dataset...")
        print(f"Dataset will be stored in: {self.data_dir}")
        
        if self.dataset_name == 'mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            self.x_train = self.x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            self.x_test = self.x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            self.input_shape = (28, 28, 1)
            self.num_classes = 10
        elif self.dataset_name == 'cifar10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
            self.x_train = self.x_train.astype('float32') / 255.0
            self.x_test = self.x_test.astype('float32') / 255.0
            self.input_shape = (32, 32, 3)
            self.num_classes = 10
        
        self.create_validation_split(validation_size=5000)

        self.y_train_oneshot = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_val_oneshot = tf.keras.utils.to_categorical(self.y_val, self.num_classes)
        self.y_test_oneshot = tf.keras.utils.to_categorical(self.y_test, self.num_classes)

        print(f"Dataset: {self.dataset_name}")
        print(f"Training set: {self.x_train.shape} with {len(self.x_train)} samples")
        print(f"Validation set: {self.x_val.shape} with {len(self.x_val)} samples")
        print(f"Test set: {self.x_test.shape} with {len(self.x_test)} samples")
        print(f"Number of classes: {self.num_classes}")

    def create_validation_split(self, validation_size=5000):
        """
        Create a validation split from the training data
        
        Args:
            validation_size (int): Size of the validation split
        """

        indices = np.arange(len(self.x_train))
        np.random.shuffle(indices)
        val_indices = indices[:validation_size]
        train_indices = indices[validation_size:]

        self.x_val = self.x_train[val_indices]
        self.y_val = self.y_train[val_indices]

        self.x_train = self.x_train[train_indices]
        self.y_train = self.y_train[train_indices]

    def prepare_subset(self, percentage):
        """
        Prepare a subset of the training data
        
        Args:
            percentage (int): Percentage of the training data to use

        Returns:
            x_subset (np.array): Subset of the training data
            y_subset (np.array): Labels of the subset of the training data
            y_subset_oneshot (np.array): One-hot encoded labels of the subset of the training data
        """
        
        if percentage <= 0 or percentage > 1:
            raise ValueError("Invalid percentage. Must be between 0 and 1")
        
        num_samples = int(percentage * len(self.x_train))
        indices = np.arange(len(self.x_train))
        np.random.shuffle(indices)
        subset_indices = indices[:num_samples]

        x_subset = self.x_train[subset_indices]
        y_subset = self.y_train[subset_indices]
        y_subset_oneshot = self.y_train_oneshot[subset_indices]

        print(f"Created subset with {num_samples} samples ({percentage:.2%} of training data)")

        return x_subset, y_subset, y_subset_oneshot
    
    def add_label_noise(self, noise_percentage):
        """
        Add label noise to the training data
        
        Args:
            noise_percentage (int): Percentage of the labels to corrupt

        Returns:
            x_train (np.array): Training data
            noisy_y_train (np.array): Noisy labels
            noisy_y_train_oneshot (np.array): One-hot encoded noisy labels
            original_y_train (np.array): Original labels
            
        """
        
        if noise_percentage <= 0 or noise_percentage > 1:
            raise ValueError("Invalid percentage. Must be between 0 and 1")
        
        original_y_train = self.y_train.copy()
        noisy_y_train = self.y_train.copy()

        num_noisy = int(noise_percentage * len(self.y_train))
        noise_indices = np.random.choice(len(self.y_train), num_noisy, replace=False)

        for idx in noise_indices:
            current_class = noisy_y_train[idx]
            available_classes = list(range(self.num_classes))
            available_classes.remove(current_class)
            noisy_y_train[idx] = np.random.choice(available_classes)

        # convert back to onehot
        noisy_y_train_oneshot = tf.keras.utils.to_categorical(noisy_y_train, self.num_classes)

        print(f"Created noisy labels with {num_noisy} corrupted labels ({noise_percentage:.2%} of training data)")
        print(f"Number of labels changed: {np.sum(noisy_y_train != original_y_train)}")

        return self.x_train, noisy_y_train, noisy_y_train_oneshot, original_y_train
    
    def create_data_generator(self, x, y, batch_size=64, augment=False):
        """
        Create a data generator
        
        Args:
            x (np.array): Data
            y (np.array): Labels
            batch_size (int): Batch size
            augment (bool): Whether to apply data augmentation
        
        Returns:
            data_generator (tf.data.Dataset): Data generator
        """
        
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        if augment:
            if self.dataset_name == 'mnist':
                def augment_mnist(image, label):
                    # random mutations
                    image = tf.image.random_rotation(image, 0.1)
                    image = tf.image.random_crop(image, tf.pad(image, [[2, 2], [2, 2], [0, 0]]), size=[28, 28, 1])
                    image = tf.image.random_brightness(image, 0.1)
                    return image, label
                dataset = dataset.map(augment_mnist)
                print("Applied MNIST data augmentation (rotation, shift, brightness)")
            elif self.dataset_name == 'cifar10':
                def augment_cifar(image, label):
                    # random mutations
                    image = tf.image.random_flip_left_right(image)
                    image = tf.image.random_flip_up_down(image)
                    image = tf.image.random_crop(tf.pad(image, [[4, 4], [4, 4], [0, 0]]), size=[32, 32, 3])
                    image = tf.image.random_brightness(image, 0.1)
                    image = tf.image.random_contrast(image, 0.1)
                    return image, label
                dataset = dataset.map(augment_cifar)
                print("Applied CIFAR-10 data augmentation (flip, crop, brightness, contrast)")
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        print(f"Created data generator with batch size {batch_size}")
        return dataset
    
    def generate_adversarial_examples(self, model, x, y, method='fgsm', epsilon=0.01, num_iterations=10):
        """
        Generate adversarial examples using FGSM or IFGSM.
        
        Args:
            model (tf.keras.Model): Model to generate adversarial examples for
            x (np.array): Data
            y (np.array): Labels
            method (str): 'fgsm' or 'ifgsm'
            epsilon (float): Perturbation factor
            num_iterations (int): Number of iterations for IFGSM
        
        Returns:
            x_adv (np.array): Adversarial examples
        """
        print(f"Generating {method.upper()} adversarial examples with epsilon={epsilon}...")
        x_adv = x.copy()

        if method.lower() == 'fgsm':
            # fast grad sign method
            with tf.GradientTape() as tape:
                x_tensor = tf.convert_to_tensor(x)
                tape.watch(x_tensor)
                y_tensor = tf.convert_to_tensor(y)
                predictions = model(x_tensor)
                loss = tf.keras.losses.categorical_crossentropy(y_tensor, predictions)
            
            gradients = tape.gradient(loss, x_tensor)
            signed_grad = tf.sign(gradients)
            x_adv = x_adv + epsilon * signed_grad
            x_adv = tf.clip_by_value(x_adv, 0, 1)

        elif method.lower() == 'ifgsm':
            # iterative fast grad sign method
            x_tensor = tf.convert_to_tensor(x)
            y_tensor = tf.convert_to_tensor(y)
            alpha = epsilon / num_iterations

            for i in range(num_iterations):
                with tf.GradientTape() as tape:
                    tape.watch(x_tensor)
                    predictions = model(x_tensor)
                    loss = tf.keras.losses.categorical_crossentropy(y_tensor, predictions)
                
                gradients = tape.gradient(loss, x_tensor)
                signed_grad = tf.sign(gradients)
                x_tensor = x_tensor + alpha * signed_grad

                # clip to ensure valid img and within epsilon ball
                x_tensor = tf.clip_by_value(x_tensor, x - epsilon, x + epsilon)
                x_tensor = tf.clip_by_value(x_tensor, 0, 1)

                if (i+1) % (num_iterations // 5) == 0 or i == 0:
                    print(f"  Iteration {i+1}/{num_iterations} completed")
                
            
            x_adv = x_tensor

        else:
            raise ValueError("Invalid method. Choose from 'fgsm', 'ifgsm'")
        print(f"Generated {len(x_adv)} adversarial examples")
        return x_adv.numpy()
