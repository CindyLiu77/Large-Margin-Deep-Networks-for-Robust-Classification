from utils.adversarial import test_accuracy, test_accuracy_attack, fgsm_attack, ifgsm_attack
import argparse
import torch
import torch.nn as nn
from models.mnist_model import MNISTModel
from models.cifar_model import cifar_resnet_small, cifar_resnet_medium, cifar_resnet_large
from utils.data_loader import get_mnist_data_loader, get_cifar10_data_loader
from losses.margin_loss import SimpleLargeMarginLoss, LargeMarginLoss, MultiLayerMarginLoss, TrueMultiLayerMarginLoss
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial Evaluation')

    # Model parameters
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--model-size', type=str, default='medium', choices=['small', 'medium', 'large'], help='Size of the ResNet model to use')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], required=True, help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used for training')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for data loading')

    # Attack parameters
    parser.add_argument('--attack', type=str, choices=['fgsm', 'ifgsm'], required=True, help='Type of attack to evaluate')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Epsilon for FGSM and I-FGSM attacks')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha for I-FGSM attack')
    parser.add_argument('--num_iter', type=int, default=10, help='Number of iterations for I-FGSM attack')

    # Loss function parameters
    parser.add_argument('--loss-type', type=str, default='cross_entropy', choices=['cross_entropy', 'simple_margin', 'margin', 'multi_layer_margin', 'true_multi_layer_margin'], help='Loss function to use')
    parser.add_argument('--gamma', type=float, default=10.0, help='Margin parameter')
    parser.add_argument('--norm', type=str, default='l2', choices=['l1', 'l2', 'linf'], help='Norm type for margin computation')
    parser.add_argument('--aggregation', type=str, default='max', choices=['max', 'sum'], help='Aggregation method for margin violations')
    parser.add_argument('--layers', type=str, default=None, help='Layers to apply multi-layer margin loss')
    parser.add_argument('--vectorize', action='store_true', help='Use vectorized implementation for multi-layer margin loss')
    parser.add_argument('--top-k', type=int, default=1, help='Top k layers to consider for multi-layer margin loss')

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)



    # Initialize the model
    print(f"Loading {args.dataset} model...")
    if args.dataset == 'mnist':
        model = MNISTModel()
    elif args.dataset == 'cifar10':
        if args.model_size == 'small':
            model = cifar_resnet_small()
        elif args.model_size == 'medium':
            model = cifar_resnet_medium()
        else:
            model = cifar_resnet_large()
    else:
        raise ValueError("Unsupported dataset. Choose either 'mnist' or 'cifar10'.")
    
    # Load the model weights
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # Generate the test data loader
    print(f"Loading {args.dataset} data...")
    if args.dataset == 'mnist':
        _, _, test_loader = get_mnist_data_loader(args.batch_size)
    elif args.dataset == 'cifar10':
        _, _, test_loader = get_cifar10_data_loader(args.batch_size)
    else:
        raise ValueError("Unsupported dataset. Choose either 'mnist' or 'cifar10'.")
    
    # print out standard test accuracy
    print("Evaluating standard test accuracy...")
    clean_accuracy = test_accuracy(model, test_loader, device)
    print(f"Standard Test Accuracy: {clean_accuracy:.4f}")

    # Find the attack function, parameters, and loss fcn based on the user's choice
    print(f"Evaluating adversarial test accuracy with {args.attack} attack...")
    if args.attack == 'fgsm':
        attack_fn = fgsm_attack
        attack_params = {'epsilon': args.epsilon}
    elif args.attack == 'ifgsm':
        attack_fn = ifgsm_attack
        attack_params = {
            'epsilon': args.epsilon,
            'alpha': args.alpha,
            'num_iter': args.num_iter
        }
    else:
        raise ValueError("Unsupported attack Choose either 'fgsm' or 'ifgsm'.")
    
    #choose loss
    if args.loss_type == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_type == 'simple_margin':
        loss_fn = SimpleLargeMarginLoss(gamma=args.gamma, aggregation=args.aggregation)
    elif args.loss_type == 'margin':
        loss_fn = LargeMarginLoss(gamma=args.gamma, norm=args.norm, 
                                  aggregation=args.aggregation)
    elif args.loss_type == 'multi_layer_margin':
        if not args.layers:
            raise ValueError("Must specify layers for multi_layer_margin loss")
        layers = [int(layer.strip()) for layer in args.layers.split(',')]
        loss_fn = MultiLayerMarginLoss(layers=layers, gamma=args.gamma, 
                                        norm=args.norm, aggregation=args.aggregation)
    elif args.loss_type == 'true_multi_layer_margin':
        if not args.layers:
            raise ValueError("Must specify layers for true_multi_layer_margin loss")
        layers = [int(layer.strip()) for layer in args.layers.split(',')]
        loss_fn = TrueMultiLayerMarginLoss(layers=layers, gamma=args.gamma, 
                                        norm=args.norm, aggregation=args.aggregation, vectorize=args.vectorize, top_k=args.top_k)
    else:
        raise ValueError("Unsupported loss type. Choose either 'cross_entropy', 'simple_margin', 'margin', 'multi_layer_margin', or 'true_multi_layer_margin'.")
     
    # Evaluate the model on adversarial examples generated by the chosen attack
    adversarial_accuracy = test_accuracy_attack(model, test_loader, args.dataset, loss_fn, attack_fn, attack_params, device)
    print(f"Adversarial Test Accuracy ({args.attack}): {adversarial_accuracy:.4f}")
    
    
    # Save the results to a file
    results = {
        'clean_accuracy': clean_accuracy,
        'adversarial_accuracy': adversarial_accuracy,
        'attack': args.attack,
        'epsilon': args.epsilon,
        'alpha': args.alpha,
        'num_iter': args.num_iter,
        'loss_type': args.loss_type,
        'gamma': args.gamma,
        'norm': args.norm,
        'aggregation': args.aggregation,
        'layers': args.layers,
        'vectorize': args.vectorize,
        'top_k': args.top_k
    }
    
    results_file = f"adversarial_results/{args.dataset}_{args.loss_type}_{args.attack}_results.txt"
    with open(results_file, 'w') as f:
        f.write(str(results))
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()

