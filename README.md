# Large Margin Deep Networks

This repository implements the Large Margin Deep Networks described in the paper "Large Margin Deep Networks for Classification" by Elsayed et al.

## Usage

The training scripts provide a flexible command-line interface to customize experiments:

### Loss Functions

The implementation supports several loss functions:

- `cross_entropy`: Standard cross-entropy loss (baseline)
- `simple_margin`: A simplified margin-based loss
- `margin`: Full large margin loss implementation
- `multi_layer_margin`: Margin applied at multiple network layers
- `true_multi_layer_margin`: Most faithful recreation of the methodology used in the paper, see TrueMultiLayerMargin class for more details

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--loss-type` | Type of loss function | `cross_entropy` |
| `--gamma` | Margin parameter for margin losses | `10.0` |
| `--norm` | Norm type for margin computation | `l2` |
| `--aggregation` | How to aggregate margins (`max` or `sum`) | `max` |
| `--batch-size` | Batch size | `64` (MNIST), `128` (CIFAR) |
| `--epochs` | Number of training epochs | `10` |
| `--lr` | Learning rate | `0.01` |
| `--momentum` | Momentum for SGD | `0.9` |
| `--weight-decay` | Weight decay for regularization | `5e-4` |
| `--optimizer` | Optimizer type | `adam` (MNIST), `sgd` (CIFAR) |
| `--noisy-labels` | Fraction of labels to corrupt | `0.0` |
| `--data-fraction` | Fraction of training data to use | `1.0` |
| `--layers` | Comma-separated list of layer indices | `` |
| `--seed` | Random seed | `42` |
| `--mixed-precision` | Enable mixed precision training (FP16) | `False` |
| `--top-kTop` | K incorrect classes to consider for margin loss | `None` |
| `--vectorize` | Use vectorized implementation for margin loss | `False` |
| `--verbose` | Enable detailed timing information | `False` |

*Note: for MNIST the layers range from 0-5, and for CIFAR the layers range []. OMit spaces when specifying multiple layers using the `--layers` flag

## Example Commands

### Training with Different Loss Functions

```bash
# Cross-entropy (baseline)
python -m train_mnist.py --loss-type cross_entropy

# Full margin loss with L2 norm
python -m train_mnist.py --loss-type margin --gamma 10.0 --norm l2

# Multi-layer margin loss (input, middle, and output layers)
python -m train_mnist.py --loss-type multi_layer_margin --gamma 10.0 --layers "0,3,5"

# True Multi-layer margin loss (all layers mnist, vectorized, top 3 incorrect classes, mixed precision)
python -m training.train_mnist --loss-type true_multi_layer_margin --norm l2 --gamma 1.0 --layers "0,1,2,3,4,5" --batch-size 128 --lr 
0.003 --mixed-precision --verbose --vectorize --top-k 3
```

### Experiments with Noisy Labels

To train models with different levels of label noise:

```bash
# 20% corrupted labels
python -m train_cifar.py --loss-type margin --gamma 15.0 --noisy-labels 0.2

# 50% corrupted labels
python -m train_cifar.py --loss-type margin --gamma 15.0 --noisy-labels 0.5

# 80% corrupted labels
python -m train_cifar.py --loss-type margin --gamma 15.0 --noisy-labels 0.8
```

### Experiments with Limited Data

To train models with reduced amounts of training data:

```bash
# 10% of training data
python -m train_cifar.py --loss-type margin --gamma 15.0 --data-fraction 0.1

# 1% of training data
python -m train_cifar.py --loss-type margin --gamma 15.0 --data-fraction 0.01

# 0.1% of training data
python -m train_cifar.py --loss-type margin --gamma 15.0 --data-fraction 0.001
```
## Visualizations

### Margin Boundaries
** WORK IN PROGRESS

### UMAP/TSNE

To generate UMAP/tSNE plots, use the visualize.py file, with the `.pth` file for the model argument, the `--method` flag to designate between umap and tsne, and the `--num-samples` flag to choose how many :

`python -m training.visualize --models checkpoints/mnist_model_margin.pth --method umap --num-samples 1000`

Note: the mnist dataset has ~10,000 samples

## Adversarial Robustness Testing

To evaluate a model's robustness to adversarial examples, you can use the adversarial utilities:

```bash
python3 -m eval.adversarial_eval --model "checkpoints/mnist_model_multi_layer_margin.pth" --dataset 'mnist' --batch-size 128 --attack 'fgsm' --norm l2 --gamma 1.0 --layers "0,1,2,3,4,5"  --top-k 3 --vectorize --epsilon 0.3 --loss-type 'multi_layer_margin'               
```

The parameters for the large part are consitent with the parameters needed for the loss.

### Summary of Parameters

#### Model Parameters
--model           (str, required)       Path to the model file.
--model-size      (str, default='medium', choices=['small', 'medium', 'large'])
                                     Size of the ResNet model to use.

#### Dataset Parameters
--dataset         (str, required, choices=['mnist', 'cifar10'])
                                     Dataset to use.
--seed            (int, default=42)   Random seed used for training.
--batch-size      (int, default=64)   Batch size for data loading.

#### Attack Parameters
--attack          (str, required, choices=['fgsm', 'ifgsm'])
                                     Type of attack to evaluate.
--epsilon         (float, default=0.3)
                                     Epsilon for FGSM and I-FGSM attacks.
--alpha           (float, default=0.01)
                                     Alpha (step size) for I-FGSM attack.
--num_iter        (int, default=10)   Number of iterations for I-FGSM attack.

#### Loss Function Parameters
--loss-type       (str, default='cross_entropy',
                  choices=['cross_entropy', 'simple_margin', 'margin', 'multi_layer_margin', 'true_multi_layer_margin'])
                                     Loss function to use.
--gamma           (float, default=10.0)
                                     Margin parameter for margin-based losses.
--norm            (str, default='l2', choices=['l1', 'l2', 'linf'])
                                     Norm type for margin computation.
--aggregation     (str, default='max', choices=['max', 'sum'])
                                     Aggregation method for margin violations.
--layers          (str, default=None) Layers to apply multi-layer margin loss.
--vectorize       (flag)              Use vectorized implementation for multi-layer margin loss.
--top-k           (int, default=1)    Top k layers to consider for multi-layer margin loss.

## References

- [Large Margin Deep Networks for Classification](https://arxiv.org/abs/1803.05598) - Elsayed et al., NeurIPS 2018
