import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLargeMarginLoss(nn.Module):
    """
    Easier to run + inital implementation of Large Margin Loss
    """
    def __init__(self, gamma=10.0, aggregation="max"):
        """
        Parameters:
        -----------
        gamma: float
            Margin parameter
        aggregation: str
            Aggregation type for incorrect classes ('max' or 'sum')
        """
        super(SimpleLargeMarginLoss, self).__init__()
        self.gamma = gamma
        self.aggregation = aggregation
    
    def forward(self, logits, targets):
        """
        Forward pass of the simplified large margin loss
        
        Parameters:
        -----------
        logits: torch.Tensor
            Model output logits (batch_size, num_classes)
        targets: torch.Tensor
            Ground truth class indices (batch_size,)
            
        Returns:
        --------
        torch.Tensor
            Loss value
        """
        batch_size, num_classes = logits.size()
        device = logits.device
        
        targets_one_hot = F.one_hot(targets, num_classes).float()
        correct_class_scores = torch.sum(logits * targets_one_hot, dim=1, keepdim=True)
        
        # Gamma + (incorrect_score - correct_score)
        margin_violations = self.gamma + logits - correct_class_scores

        mask = 1.0 - targets_one_hot
        margin_violations = margin_violations * mask
        margin_violations = F.relu(margin_violations)
        
        # Choose aggregation type
        if self.aggregation == "max":
            loss = torch.max(margin_violations, dim=1)[0]
        else:
            loss = torch.sum(margin_violations, dim=1)
        return torch.mean(loss)


class LargeMarginLoss(nn.Module):
    def __init__(self, gamma=10.0, norm="l2", aggregation="max", epsilon=1e-6):
        """
        Parameters:
        -----------
        gamma: float
            Margin parameter
        norm: str
            Norm type for computing distance ('l1', 'l2', 'linf')
        aggregation: str
            Aggregation type for incorrect classes ('max' or 'sum')
        epsilon: float
            Small value to avoid division by zero
        """
        super(LargeMarginLoss, self).__init__()
        self.gamma = gamma
        self.norm = norm
        self.aggregation = aggregation
        self.epsilon = epsilon
    
    def compute_dual_norm(self, grad_diff):
        """
        Compute the dual norm of gradient differences
        
        Parameters:
        -----------
        grad_diff: torch.Tensor
            Gradient difference tensor
            
        Returns:
        --------
        torch.Tensor
            Dual norm of gradient differences
        """
        if self.norm == "l1":
            # Dual of l1 is linf
            return torch.amax(torch.abs(grad_diff), dim=list(range(1, grad_diff.dim())))
        elif self.norm == "l2":
            # Dual of l2 is l2
            return torch.sqrt(torch.sum(grad_diff**2, dim=list(range(1, grad_diff.dim()))) + self.epsilon)
        elif self.norm == "linf":
            # Dual of linf is l1
            return torch.sum(torch.abs(grad_diff), dim=list(range(1, grad_diff.dim())))
        else:
            raise ValueError(f"Unsupported norm type: {self.norm}")
    
    def forward(self, logits, targets, layer_activations=None):
        """
        Forward pass of the large margin loss
        
        Parameters:
        -----------
        logits: torch.Tensor
            Model output logits (batch_size, num_classes)
        targets: torch.Tensor
            Ground truth class indices (batch_size,)
        layer_activations: torch.Tensor or None
            Layer activations to compute margin on. If None, defaults to input.
            
        Returns:
        --------
        torch.Tensor
            Loss value
        """
        batch_size, num_classes = logits.size()
        device = logits.device
        
        # default to simple if no layers arg
        if layer_activations is None:
            return SimpleLargeMarginLoss(self.gamma, self.aggregation)(logits, targets)
        
        if not layer_activations.requires_grad:
            layer_activations.requires_grad_(True)
        

        targets_one_hot = F.one_hot(targets, num_classes).float().to(device)
        correct_class_scores = torch.sum(logits * targets_one_hot, dim=1)
        loss = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            target = targets[i]
            correct_score = correct_class_scores[i]
            max_violation = torch.tensor(-float('inf'), device=device)
            
            for j in range(num_classes):
                if j == target:
                    continue
                
                # Gradients of correct class
                grad_correct = torch.autograd.grad(
                    correct_score, layer_activations, 
                    create_graph=True, retain_graph=True, allow_unused=True
                )[0][i]
                
                # Grad of incorrect class
                grad_incorrect = torch.autograd.grad(
                    logits[i, j], layer_activations, 
                    create_graph=True, retain_graph=True, allow_unused=True
                )[0][i]
                
                grad_diff = grad_incorrect - grad_correct
                grad_diff_norm = self.compute_dual_norm(grad_diff.unsqueeze(0)).squeeze() + self.epsilon # DUal norm
                score_diff = logits[i, j] - correct_score
                
                # margin violation
                violation = self.gamma + score_diff / grad_diff_norm
                violation = F.relu(violation)
                
                if self.aggregation == "max":
                    max_violation = torch.max(max_violation, violation)
                else:
                    if torch.isinf(max_violation):
                        max_violation = violation
                    else:
                        max_violation += violation
            loss += max_violation
        return loss / batch_size


class MultiLayerMarginLoss(nn.Module):
    """
    Implementation of Large Margin Loss that can be applied to multiple layers.
    """
    def __init__(self, layers, gamma=10.0, norm="l2", aggregation="max", epsilon=1e-6):
        """
        Parameters:
        -----------
        layers: list of int
            Indices of layers to apply margin on (-1 for output, 0 for input, etc.)
        gamma: float or list of float
            Margin parameter(s) for each layer
        norm: str or list of str
            Norm type(s) for computing distance
        aggregation: str
            Aggregation type for incorrect classes
        epsilon: float
            Small value to avoid division by zero
        """
        super(MultiLayerMarginLoss, self).__init__()
        self.layers = layers
        
        # args handlers
        if isinstance(gamma, (int, float)):
            self.gamma = [gamma] * len(layers)
        else:
            self.gamma = gamma
            
        if isinstance(norm, str):
            self.norm = [norm] * len(layers)
        else:
            self.norm = norm
            
        self.aggregation = aggregation
        self.epsilon = epsilon
        
        # margin loss per layer
        self.layer_losses = nn.ModuleList([
            LargeMarginLoss(g, n, aggregation, epsilon)
            for g, n in zip(self.gamma, self.norm)
        ])
    
    def forward(self, logits, targets, activations):
        """
        Forward pass of the multi-layer large margin loss
        
        Parameters:
        -----------
        logits: torch.Tensor
            Model output logits
        targets: torch.Tensor
            Ground truth class indices
        activations: list of torch.Tensor
            List of activations from different layers
            
        Returns:
        --------
        torch.Tensor
            Loss value
        """
        if len(activations) < max(abs(layer_idx) for layer_idx in self.layers) + 1:
            raise ValueError(f"Not enough activations provided. Need at least {max(abs(layer_idx) for layer_idx in self.layers) + 1}, got {len(activations)}")

        total_loss = 0.0
        
        # Compute loss for each layer
        for i, layer_idx in enumerate(self.layers):
            layer_loss = self.layer_losses[i]
            layer_activation = activations[layer_idx]
            total_loss += layer_loss(logits, targets, layer_activation)
            
        return total_loss / len(self.layers)