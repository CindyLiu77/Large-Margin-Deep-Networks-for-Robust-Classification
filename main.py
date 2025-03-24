import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import time
import argparse
from losses.large_margin_loss import LargeMarginLoss

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset_loader import DatasetLoader
os.environ['KERAS_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

np.random.seed(42)
tf.random.set_seed(42)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
models_dir = os.path.join(results_dir, "models")
figures_dir = os.path.join(results_dir, "figures")
logs_dir = os.path.join(results_dir, "logs")
for directory in [data_dir, results_dir, models_dir, figures_dir, logs_dir]:
    os.makedirs(directory, exist_ok=True)

is_keras3 = int(tf.keras.__version__.split('.')[0]) >= 3

print(f"Using Keras version: {tf.keras.__version__}")
print(f"Is Keras 3 or higher: {is_keras3}")

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


def create_mnist_model():
    """
    Create a simple MNIST model as described in the paper using Functional API.
    4 hidden layers: 2 convolutional followed by 2 fully connected.
    """
    inputs = tf.keras.layers.Input(shape=(28, 28, 1), name='input_layer')
    
    # First convolutional layer
    x = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', name='conv1')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    
    # Second convolutional layer
    x = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', name='conv2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    
    # Flatten for fully connected layers
    x = tf.keras.layers.Flatten(name='flatten')(x)
    
    # First fully connected layer
    x = tf.keras.layers.Dense(512, activation='relu', name='fc1')(x)
    
    # Second fully connected layer
    x = tf.keras.layers.Dense(512, activation='relu', name='fc2')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cifar_resnet(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a simplified ResNet model for CIFAR-10.
    """
    def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
        shortcut = x
        if conv_shortcut:
            shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.add([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name='input_layer')
    
    # Initial convolution
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # First residual block
    x = residual_block(x, 64, conv_shortcut=True)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    # Second residual block
    x = residual_block(x, 128, stride=2, conv_shortcut=True)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    # Third residual block
    x = residual_block(x, 256, stride=2, conv_shortcut=True)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # Global average pooling and output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_model(model_type, loss_type, dataset_name, epochs=20, batch_size=64, 
                norm_type='l2', gamma=10.0, margin_layers=None, 
                noise_ratio=0.0, data_ratio=1.0, 
                augment=False, save_model=True):
    """
    Train a model with specified parameters.
    
    Args:
        model_type (str): 'mnist' or 'cifar'
        loss_type (str): 'cross_entropy', 'hinge', or 'margin'
        dataset_name (str): 'mnist' or 'cifar10'
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        norm_type (str): 'l1', 'l2', or 'linf' (for margin loss)
        gamma (float): Margin value (for margin loss)
        margin_layers (list): List of layer names to apply margin (for margin loss)
        noise_ratio (float): Ratio of noisy labels
        data_ratio (float): Ratio of training data to use
        augment (bool): Whether to use data augmentation
        save_model (bool): Whether to save the trained model
        
    Returns:
        tuple: (model, history)
    """
    print(f"\n{'='*80}")
    print(f"TRAINING {model_type.upper()} MODEL WITH {loss_type.upper()} LOSS")
    if loss_type == 'margin':
        print(f"MARGIN TYPE: {norm_type}, GAMMA: {gamma}")
        print(f"MARGIN LAYERS: {margin_layers}")
    print(f"DATASET: {dataset_name.upper()}, NOISE: {noise_ratio*100}%, DATA RATIO: {data_ratio*100}%")
    print(f"EPOCHS: {epochs}, BATCH SIZE: {batch_size}, AUGMENTATION: {augment}")
    print(f"{'='*80}\n")

    # Load dataset
    dataset = DatasetLoader(dataset_name=dataset_name, data_dir=data_dir)
    
    # Apply noise to labels if specified
    if noise_ratio > 0:
        x_train, y_train, y_train_onehot, _ = dataset.add_label_noise(noise_ratio)
    else:
        x_train, y_train, y_train_onehot = dataset.x_train, dataset.y_train, dataset.y_train_oneshot
    
    # Use subset of data if specified
    if data_ratio < 1.0:
        x_train, y_train, y_train_onehot = dataset.prepare_subset(data_ratio)
    
    # Create data generators
    train_generator = dataset.create_data_generator(
        x_train, y_train_onehot, batch_size=batch_size, augment=augment
    )
    val_generator = dataset.create_data_generator(
        dataset.x_val, dataset.y_val_oneshot, batch_size=batch_size
    )
    test_generator = dataset.create_data_generator(
        dataset.x_test, dataset.y_test_oneshot, batch_size=batch_size
    )
    
    # Create base model
    if model_type == 'mnist':
        base_model = create_mnist_model()
    elif model_type == 'cifar':
        base_model = create_cifar_resnet()
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Setup callbacks
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f"{model_type}_{loss_type}"
    if loss_type == 'margin':
        model_name += f"_{norm_type}_gamma{gamma}"
    model_name += f"_noise{int(noise_ratio*100)}_data{int(data_ratio*100)}"
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(logs_dir, f"{model_name}_{current_time}"),
        histogram_freq=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        mode='max'
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(models_dir, f"{model_name}_best.keras"),
        monitor='val_accuracy',
        save_best_only=True
    )

    # Set up model based on loss type
    if loss_type == 'cross_entropy':
        model = base_model
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    elif loss_type == 'hinge':
        model = base_model
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalHinge(),
            metrics=['accuracy']
        )
    elif loss_type == 'margin':
        # Default layers if not specified
        if margin_layers is None:
            # Get available layers
            available_layers = [layer.name for layer in base_model.layers]
            print(f"Available layers in base model: {available_layers}")
            
            if model_type == 'mnist':
                # Try to use input layer, a middle hidden layer, and output
                if 'input_layer' in available_layers:
                    margin_layers = ['input_layer', 'fc1', 'fc2', 'output']
                else:
                    # Use first, middle and last layer
                    margin_layers = [
                        available_layers[0], 
                        available_layers[len(available_layers)//2], 
                        available_layers[-1]
                    ]
            else:  # cifar
                # Select evenly spaced layers as mentioned in the paper
                all_layers = [layer.name for layer in base_model.layers]
                margin_layers = []
                
                # Add input layer if it exists
                if 'input_layer' in all_layers:
                    margin_layers.append('input_layer')
                else:
                    margin_layers.append(all_layers[0])
                    
                # Add some intermediate layers
                indices = np.linspace(1, len(all_layers)-2, 3, dtype=int)
                for idx in indices:
                    margin_layers.append(all_layers[idx])
                margin_layers.append(all_layers[-1])
        
        print(f"Using margin layers: {margin_layers}")
        
        model = LargeMarginLoss(
            base_model=base_model,
            margin_layers=margin_layers,
            gamma=gamma,
            norm=norm_type,
            epsilon=1e-6,
            cross_entropy_weight=1.0,
            aggregation='max' if model_type == 'mnist' else 'sum',
            top_k_classes=9 if dataset_name == 'mnist' else 9,
            use_approximation=True
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=model.compute_large_margin_loss,
            metrics=['accuracy']
        )
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[tensorboard_callback, early_stopping, model_checkpoint]
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    
    # Save model if specified
    if save_model:
        model.save(os.path.join(models_dir, f"{model_name}_final.keras"))
        print(f"Model saved to {os.path.join(models_dir, f'{model_name}_final.keras')}")
    
    return model, history

def plot_training_history(histories, labels, title, filename):
    """
    Plot training and validation metrics for multiple models.
    
    Args:
        histories (list): List of history objects
        labels (list): List of model labels
        title (str): Plot title
        filename (str): Output filename
    """
    plt.figure(figsize=(12, 10))
    
    # Plot training & validation accuracy
    plt.subplot(2, 1, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'{labels[i]} - Training')
        plt.plot(history.history['val_accuracy'], label=f'{labels[i]} - Validation')
    
    plt.title(f'{title} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot training & validation loss
    plt.subplot(2, 1, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'{labels[i]} - Training')
        plt.plot(history.history['val_loss'], label=f'{labels[i]} - Validation')
    
    plt.title(f'{title} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename))
    plt.close()
    print(f"Training history plot saved to {os.path.join(figures_dir, filename)}")

def evaluate_adversarial_robustness(models, model_labels, dataset_name, 
                                   epsilons=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], 
                                   methods=['fgsm', 'ifgsm'], 
                                   batch_size=64, num_iterations=10):
    """
    Evaluate models on adversarial examples.
    
    Args:
        models (list): List of trained models
        model_labels (list): List of model labels
        dataset_name (str): Dataset name
        epsilons (list): List of epsilon values for adversarial perturbations
        methods (list): List of adversarial attack methods
        batch_size (int): Batch size for evaluation
        num_iterations (int): Number of iterations for IFGSM
        
    Returns:
        dict: Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ADVERSARIAL ROBUSTNESS")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"EPSILONS: {epsilons}")
    print(f"METHODS: {methods}")
    print(f"{'='*80}\n")
    
    # Load dataset
    dataset = DatasetLoader(dataset_name=dataset_name, data_dir=data_dir)
    
    # Create data generator for test set
    test_generator = dataset.create_data_generator(
        dataset.x_test, dataset.y_test_oneshot, batch_size=batch_size
    )
    
    # Select a subset of test data for generating adversarial examples
    # to make the evaluation faster
    num_samples = 1000
    indices = np.random.choice(len(dataset.x_test), num_samples, replace=False)
    x_subset = dataset.x_test[indices]
    y_subset = dataset.y_test_oneshot[indices]
    
    # Dictionary to store results
    results = {method: {model_label: [] for model_label in model_labels} for method in methods}
    
    # For each attack method
    for method in methods:
        print(f"\nEvaluating {method.upper()} attack:")
        
        # For each epsilon value
        for epsilon in epsilons:
            print(f"\n  Epsilon = {epsilon}:")
            
            # Generate adversarial examples using the first model
            # We're using white-box attack against the first model,
            # and evaluating all models on these examples
            print(f"  Generating adversarial examples using {model_labels[0]}...")
            x_adv = dataset.generate_adversarial_examples(
                models[0], x_subset, y_subset, 
                method=method, epsilon=epsilon, 
                num_iterations=num_iterations
            )
            
            # Evaluate each model on the adversarial examples
            for i, (model, model_label) in enumerate(zip(models, model_labels)):
                print(f"  Evaluating {model_label}...")
                score = model.evaluate(
                    tf.data.Dataset.from_tensor_slices((x_adv, y_subset)).batch(batch_size),
                    verbose=0
                )[1]  # Get accuracy
                results[method][model_label].append(score)
                print(f"  {model_label} accuracy: {score*100:.2f}%")
    
    # Plot results
    for method in methods:
        plt.figure(figsize=(10, 6))
        for model_label in model_labels:
            plt.plot(epsilons, results[method][model_label], marker='o', label=model_label)
        
        plt.title(f'{method.upper()} Attack - {dataset_name.upper()}')
        plt.xlabel('Epsilon')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, f'adversarial_{method}_{dataset_name}.png'))
        plt.close()
        print(f"Adversarial robustness plot saved to {os.path.join(figures_dir, f'adversarial_{method}_{dataset_name}.png')}")
    
    return results

def evaluate_noisy_labels(model_type, dataset_name, loss_types, norm_types=None, 
                         noise_ratios=[0.0, 0.2, 0.4, 0.6, 0.8], 
                         epochs=10, batch_size=64):
    """
    Evaluate models on noisy labels.
    
    Args:
        model_type (str): 'mnist' or 'cifar'
        dataset_name (str): Dataset name
        loss_types (list): List of loss types
        norm_types (list): List of norm types for margin loss
        noise_ratios (list): List of noise ratios
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        dict: Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING NOISY LABELS")
    print(f"MODEL: {model_type.upper()}, DATASET: {dataset_name.upper()}")
    print(f"NOISE RATIOS: {noise_ratios}")
    print(f"{'='*80}\n")
    
    # Dictionary to store results
    results = {loss_type: [] for loss_type in loss_types}
    if 'margin' in loss_types and norm_types is not None:
        results = {loss_type if loss_type != 'margin' else f'margin_{norm_type}': [] 
                  for loss_type in loss_types for norm_type in (norm_types if loss_type == 'margin' else [None])}
    
    # Train and evaluate models for each noise ratio
    for noise_ratio in noise_ratios:
        print(f"\nTraining with noise ratio: {noise_ratio}")
        
        # For each loss type
        for loss_type in loss_types:
            if loss_type == 'margin' and norm_types is not None:
                # For each norm type
                for norm_type in norm_types:
                    print(f"\nTraining {loss_type} model with {norm_type} norm...")
                    model, _ = train_model(
                        model_type=model_type,
                        loss_type=loss_type,
                        dataset_name=dataset_name,
                        epochs=epochs,
                        batch_size=batch_size,
                        norm_type=norm_type,
                        noise_ratio=noise_ratio,
                        save_model=False
                    )
                    
                    # Evaluate on clean test set
                    dataset = DatasetLoader(dataset_name=dataset_name, data_dir=data_dir)
                    test_generator = dataset.create_data_generator(
                        dataset.x_test, dataset.y_test_oneshot, batch_size=batch_size
                    )
                    _, test_acc = model.evaluate(test_generator)
                    results[f'margin_{norm_type}'].append(test_acc)
                    print(f"Test accuracy: {test_acc*100:.2f}%")
            else:
                print(f"\nTraining {loss_type} model...")
                model, _ = train_model(
                    model_type=model_type,
                    loss_type=loss_type,
                    dataset_name=dataset_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    noise_ratio=noise_ratio,
                    save_model=False
                )
                
                # Evaluate on clean test set
                dataset = DatasetLoader(dataset_name=dataset_name, data_dir=data_dir)
                test_generator = dataset.create_data_generator(
                    dataset.x_test, dataset.y_test_oneshot, batch_size=batch_size
                )
                _, test_acc = model.evaluate(test_generator)
                results[loss_type].append(test_acc)
                print(f"Test accuracy: {test_acc*100:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for loss_type, accuracies in results.items():
        plt.plot(noise_ratios, accuracies, marker='o', label=loss_type)
    
    plt.title(f'Noisy Labels - {dataset_name.upper()}')
    plt.xlabel('Noise Ratio')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f'noisy_labels_{dataset_name}.png'))
    plt.close()
    print(f"Noisy labels plot saved to {os.path.join(figures_dir, f'noisy_labels_{dataset_name}.png')}")
    
    return results

def evaluate_generalization(model_type, dataset_name, loss_types, norm_types=None, 
                           data_ratios=[1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0025, 0.00125], 
                           epochs=10, batch_size=64):
    """
    Evaluate models on limited data.
    
    Args:
        model_type (str): 'mnist' or 'cifar'
        dataset_name (str): Dataset name
        loss_types (list): List of loss types
        norm_types (list): List of norm types for margin loss
        data_ratios (list): List of data ratios
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        dict: Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING GENERALIZATION")
    print(f"MODEL: {model_type.upper()}, DATASET: {dataset_name.upper()}")
    print(f"DATA RATIOS: {data_ratios}")
    print(f"{'='*80}\n")
    
    # Dictionary to store results
    results = {loss_type: [] for loss_type in loss_types}
    if 'margin' in loss_types and norm_types is not None:
        results = {loss_type if loss_type != 'margin' else f'margin_{norm_type}': [] 
                  for loss_type in loss_types for norm_type in (norm_types if loss_type == 'margin' else [None])}
    
    # Train and evaluate models for each data ratio
    for data_ratio in data_ratios:
        print(f"\nTraining with data ratio: {data_ratio}")
        
        # For each loss type
        for loss_type in loss_types:
            if loss_type == 'margin' and norm_types is not None:
                # For each norm type
                for norm_type in norm_types:
                    print(f"\nTraining {loss_type} model with {norm_type} norm...")
                    model, _ = train_model(
                        model_type=model_type,
                        loss_type=loss_type,
                        dataset_name=dataset_name,
                        epochs=epochs,
                        batch_size=batch_size,
                        norm_type=norm_type,
                        data_ratio=data_ratio,
                        save_model=False
                    )
                    
                    # Evaluate on test set
                    dataset = DatasetLoader(dataset_name=dataset_name, data_dir=data_dir)
                    test_generator = dataset.create_data_generator(
                        dataset.x_test, dataset.y_test_oneshot, batch_size=batch_size
                    )
                    _, test_acc = model.evaluate(test_generator)
                    results[f'margin_{norm_type}'].append(test_acc)
                    print(f"Test accuracy: {test_acc*100:.2f}%")
            else:
                print(f"\nTraining {loss_type} model...")
                model, _ = train_model(
                    model_type=model_type,
                    loss_type=loss_type,
                    dataset_name=dataset_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    data_ratio=data_ratio,
                    save_model=False
                )
                
                # Evaluate on test set
                dataset = DatasetLoader(dataset_name=dataset_name, data_dir=data_dir)
                test_generator = dataset.create_data_generator(
                    dataset.x_test, dataset.y_test_oneshot, batch_size=batch_size
                )
                _, test_acc = model.evaluate(test_generator)
                results[loss_type].append(test_acc)
                print(f"Test accuracy: {test_acc*100:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for loss_type, accuracies in results.items():
        plt.plot(data_ratios, accuracies, marker='o', label=loss_type)
    
    plt.title(f'Generalization - {dataset_name.upper()}')
    plt.xlabel('Data Ratio')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.savefig(os.path.join(figures_dir, f'generalization_{dataset_name}.png'))
    plt.close()
    print(f"Generalization plot saved to {os.path.join(figures_dir, f'generalization_{dataset_name}.png')}")
    
    return results

def main():
    """
    Main function to run experiments.
    """
    parser = argparse.ArgumentParser(description='Test Large Margin Loss')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use (mnist or cifar10)')
    parser.add_argument('--experiment', type=str, default='all', 
                      help='Experiment to run (train, adversarial, noisy, generalization, all)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--quick', action='store_true', help='Run quick experiments (fewer epochs, fewer points)')
    args = parser.parse_args()
    
    # Adjust parameters for quick experiments
    if args.quick:
        args.epochs = 2
        
    if args.dataset == 'mnist':
        model_type = 'mnist'
    elif args.dataset == 'cifar10':
        model_type = 'cifar'
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    # Run experiments
    if args.experiment == 'train' or args.experiment == 'all':
        # Train models
        print("\nTraining baseline models...")
        ce_model, ce_history = train_model(
            model_type=model_type,
            loss_type='cross_entropy',
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        hinge_model, hinge_history = train_model(
            model_type=model_type,
            loss_type='hinge',
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("\nTraining margin models...")
        margin_models = []
        margin_histories = []
        
        for norm_type in ['l1', 'l2', 'linf']:
            model, history = train_model(
                model_type=model_type,
                loss_type='margin',
                dataset_name=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                norm_type=norm_type
            )
            margin_models.append(model)
            margin_histories.append(history)
        
        # Plot training history
        plot_training_history(
            histories=[ce_history, hinge_history] + margin_histories,
            labels=['Cross-Entropy', 'Hinge', 'Margin-L1', 'Margin-L2', 'Margin-Linf'],
            title=f'{args.dataset.upper()} Training',
            filename=f'training_history_{args.dataset}.png'
        )
    
    if args.experiment == 'adversarial' or args.experiment == 'all':
        # Load models if they exist, otherwise train them
        models = []
        model_labels = ['Cross-Entropy', 'Margin-L1', 'Margin-L2', 'Margin-Linf']
        
        for i, config in enumerate([
            ('cross_entropy', None),
            ('margin', 'l1'),
            ('margin', 'l2'),
            ('margin', 'linf')
        ]):
            loss_type, norm_type = config
            model_name = f"{model_type}_{loss_type}"
            if loss_type == 'margin':
                model_name += f"_{norm_type}_gamma10"
            model_name += "_noise0_data100_final.keras"
            model_path = os.path.join(models_dir, model_name)
            
            if os.path.exists(model_path):
                print(f"Loading model {model_labels[i]} from {model_path}")
                if loss_type == 'cross_entropy':
                    model = tf.keras.models.load_model(model_path)
                else:  # margin
                    # For margin models, we need to recreate the base model and use it
                    base_model = create_mnist_model() if model_type == 'mnist' else create_cifar_resnet()
                    model = LargeMarginLoss(
                        base_model=base_model,
                        norm=norm_type,
                        gamma=10.0
                    )
                    model.load_weights(model_path)
            else:
                print(f"Model {model_labels[i]} not found, training it...")
                if loss_type == 'cross_entropy':
                    model, _ = train_model(
                        model_type=model_type,
                        loss_type=loss_type,
                        dataset_name=args.dataset,
                        epochs=args.epochs,
                        batch_size=args.batch_size
                    )
                else:  # margin
                    model, _ = train_model(
                        model_type=model_type,
                        loss_type=loss_type,
                        dataset_name=args.dataset,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        norm_type=norm_type
                    )
            
            models.append(model)
        
        # Define epsilon values based on dataset
        if args.dataset == 'mnist':
            epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] if not args.quick else [0.1, 0.2, 0.3]
        else:  # cifar10
            epsilons = [0.01, 0.02, 0.03, 0.04, 0.05] if not args.quick else [0.01, 0.03, 0.05]
            
        # Evaluate adversarial robustness
        evaluate_adversarial_robustness(
            models=models,
            model_labels=model_labels,
            dataset_name=args.dataset,
            epsilons=epsilons,
            methods=['fgsm', 'ifgsm'] if not args.quick else ['fgsm'],
            batch_size=args.batch_size,
            num_iterations=5 if args.quick else 10
        )
    
    if args.experiment == 'noisy' or args.experiment == 'all':
        # Define noise ratios
        if args.quick:
            noise_ratios = [0.0, 0.4, 0.8]
        else:
            noise_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
            
        # Evaluate noisy labels
        evaluate_noisy_labels(
            model_type=model_type,
            dataset_name=args.dataset,
            loss_types=['cross_entropy', 'hinge', 'margin'],
            norm_types=['l1', 'l2', 'linf'],
            noise_ratios=noise_ratios,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    if args.experiment == 'generalization' or args.experiment == 'all':
        # Define data ratios
        if args.quick:
            if args.dataset == 'mnist':
                data_ratios = [0.1, 0.01, 0.001]
            else:  # cifar10
                data_ratios = [0.5, 0.1, 0.05]
        else:
            if args.dataset == 'mnist':
                data_ratios = [1.0, 0.1, 0.01, 0.001]
            else:  # cifar10
                data_ratios = [1.0, 0.5, 0.2, 0.1, 0.05]
            
        # Evaluate generalization
        evaluate_generalization(
            model_type=model_type,
            dataset_name=args.dataset,
            loss_types=['cross_entropy', 'hinge', 'margin'],
            norm_types=['l1', 'l2', 'linf'],
            data_ratios=data_ratios,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
    print("\nAll experiments completed!")


if __name__ == "__main__":
    # start_time = time.time()
    # directories = [
    #     'data',
    #     'losses',
    #     'results',
    #     'results/figures',
    #     'results/models',
    #     'results/logs',
    #     'utils'
    # ]
    
    # for directory in directories:
    #     dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)
    #         print(f"Created directory: {dir_path}")
    
    # # Run dataset loader tests
    # test_dataset_loader()
    # execution_time = time.time() - start_time
    # print(f"\nExecution completed in {execution_time:.2f} seconds")

    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")
