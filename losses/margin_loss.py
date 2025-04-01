import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLargeMarginLoss(nn.Module):
    """
    Easier to run + initial implementation of Large Margin Loss
    """
    def __init__(self, gamma=10.0, aggregation="max"):
        super(SimpleLargeMarginLoss, self).__init__()
        self.gamma = gamma
        self.aggregation = aggregation
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        batch_size, num_classes = logits.size()
        device = logits.device
        
        # Get cross entropy for stability
        ce_loss = self.cross_entropy(logits, targets)
        
        targets_one_hot = F.one_hot(targets, num_classes).float()
        correct_class_scores = torch.sum(logits * targets_one_hot, dim=1, keepdim=True)
        
        # Gamma + (incorrect_score - correct_score)
        margin_violations = self.gamma + logits - correct_class_scores
        mask = 1.0 - targets_one_hot
        margin_violations = margin_violations * mask
        margin_violations = F.relu(margin_violations)
        
        # Choose aggregation type
        if self.aggregation == "max":
            margin_loss = torch.max(margin_violations, dim=1)[0]
        else:
            margin_loss = torch.sum(margin_violations, dim=1)
        
        margin_loss = torch.mean(margin_loss)
        
        # Combined loss - start with more weight on CE for better initialization
        # Use a small weight for margin loss initially
        total_loss = 0.8 * ce_loss + 0.2 * margin_loss
        
        return total_loss


class LargeMarginLoss(nn.Module):
    """
    Improved implementation of Large Margin Loss
    """
    def __init__(self, gamma=1.0, norm="l2", aggregation="max", epsilon=1e-6):
        super(LargeMarginLoss, self).__init__()
        self.gamma = gamma
        self.norm = norm
        self.aggregation = aggregation
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss()
        # Use simple margin loss as a fallback
        self.simple_margin = SimpleLargeMarginLoss(gamma=gamma, aggregation=aggregation)
    
    def compute_dual_norm(self, grad_diff):
        """
        Compute the dual norm of gradient differences
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
        """
        # Fall back to simple margin loss if no activations provided or in eval mode
        if layer_activations is None or not self.training:
            return self.cross_entropy(logits, targets)
        
        batch_size, num_classes = logits.size()
        device = logits.device
        
        ce_loss = self.cross_entropy(logits, targets)
        
        simple_loss = self.simple_margin(logits, targets)
        
        if not layer_activations.requires_grad:
            layer_activations.requires_grad_(True)

        targets_one_hot = F.one_hot(targets, num_classes).float().to(device)
        
        correct_class_scores = torch.sum(logits * targets_one_hot, dim=1)

        margin_loss = torch.tensor(0.0, device=device)
        valid_count = 0
        
        mini_batch_size = min(8, batch_size)
        for batch_start in range(0, batch_size, mini_batch_size):
            batch_end = min(batch_start + mini_batch_size, batch_size)
            mini_batch_indices = torch.arange(batch_start, batch_end, device=device)
            
            for i in mini_batch_indices:
                i = i.item()
                target = targets[i]

                logits_without_correct = logits[i].clone()
                logits_without_correct[target] = -float('inf')
                confusing_class = torch.argmax(logits_without_correct).item()
                
                score_diff = logits[i, confusing_class] - correct_class_scores[i]
                if score_diff <= -self.gamma * 2:
                    continue
                
                try:
                    grad_correct = torch.autograd.grad(
                        correct_class_scores[i], layer_activations, 
                        create_graph=True, retain_graph=True, allow_unused=True
                    )[0]
                    
                    if grad_correct is None:
                        continue
                    
                    grad_correct = grad_correct[i]
                    
                    grad_incorrect = torch.autograd.grad(
                        logits[i, confusing_class], layer_activations, 
                        create_graph=True, retain_graph=True, allow_unused=True
                    )[0][i]
                    
                    grad_diff = grad_incorrect - grad_correct
                    grad_diff_norm = self.compute_dual_norm(grad_diff.unsqueeze(0)).squeeze() + self.epsilon
                    
                    violation = self.gamma + score_diff / grad_diff_norm
                    violation = F.relu(violation)
                    
                    margin_loss += violation
                    valid_count += 1
                        
                except Exception as e:
                    continue
        
        if valid_count == 0:
            return 0.8 * ce_loss + 0.2 * simple_loss
        
        margin_loss = margin_loss / valid_count
        total_loss = 0.6 * ce_loss + 0.2 * simple_loss + 0.2 * margin_loss
        
        return total_loss


class MultiLayerMarginLoss(nn.Module):
    """
    Improved multi-layer margin loss implementation
    """
    def __init__(self, layers, gamma=0.5, norm="l2", aggregation="max", epsilon=1e-6):
        super(MultiLayerMarginLoss, self).__init__()
        self.layers = layers
        
        if isinstance(gamma, (int, float)):
            self.gamma = [gamma * 0.5 if i > 0 else gamma for i in range(len(layers))]
        else:
            self.gamma = gamma
            
        if isinstance(norm, str):
            self.norm = [norm] * len(layers)
        else:
            self.norm = norm
            
        self.aggregation = aggregation
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss()
        
        self.simple_margin = SimpleLargeMarginLoss(gamma=gamma if isinstance(gamma, (int, float)) else gamma[0], 
                                               aggregation=aggregation)
        
        self.epoch = 0
    
    def update_epoch(self, epoch):
        """Update the current epoch - useful for adaptive weighting"""
        self.epoch = epoch
    
    def compute_dual_norm(self, grad_diff, norm_type):
        """
        Compute the dual norm of gradient differences
        """
        if norm_type == "l1":
            # Dual of l1 is linf
            return torch.amax(torch.abs(grad_diff), dim=list(range(1, grad_diff.dim())))
        elif norm_type == "l2":
            # Dual of l2 is l2
            return torch.sqrt(torch.sum(grad_diff**2, dim=list(range(1, grad_diff.dim()))) + self.epsilon)
        elif norm_type == "linf":
            # Dual of linf is l1
            return torch.sum(torch.abs(grad_diff), dim=list(range(1, grad_diff.dim())))
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
    
    def forward(self, logits, targets, activations=None):
        """
        Forward pass of the multi-layer large margin loss
        """
        # Fall back to cross-entropy if no activations provided or in eval mode
        if activations is None or not self.training:
            return self.cross_entropy(logits, targets)
        
        if len(activations) < max(abs(layer_idx) for layer_idx in self.layers) + 1:
            raise ValueError(f"Not enough activations provided. Need at least {max(abs(layer_idx) for layer_idx in self.layers) + 1}, got {len(activations)}")

        batch_size, num_classes = logits.size()
        device = logits.device
        
        ce_loss = self.cross_entropy(logits, targets)
        simple_loss = self.simple_margin(logits, targets)
        

        active_layers = [0] if self.epoch < 3 else \
                      self.layers[:min(1 + self.epoch // 2, len(self.layers))]
        
        targets_one_hot = F.one_hot(targets, num_classes).float().to(device)
        
        correct_class_scores = torch.sum(logits * targets_one_hot, dim=1)
        margin_losses = []
        
        # Start with a small subset of the batch for efficiency
        # Gradually increase as training progresses
        effective_batch_size = min(batch_size, 4 + self.epoch * 2)
        sample_indices = torch.randperm(batch_size)[:effective_batch_size]
        
        # Process each layer
        for layer_idx in active_layers:
            layer_gamma = self.gamma[self.layers.index(layer_idx)]
            layer_norm = self.norm[self.layers.index(layer_idx)]
            
            # Get activations for this layer
            layer_activation = activations[layer_idx]
            if layer_activation is None:
                continue
                
            if not layer_activation.requires_grad:
                layer_activation.requires_grad_(True)
            
            # Track valid margin computations for this layer
            layer_margin = torch.tensor(0.0, device=device)
            layer_valid_count = 0
            
            # subset of samples
            for i in sample_indices:
                i = i.item()
                target = targets[i]
                
                # Find most confusing class
                logits_without_correct = logits[i].clone()
                logits_without_correct[target] = -float('inf')
                confusing_class = torch.argmax(logits_without_correct).item()
                
                # skip if margin
                score_diff = logits[i, confusing_class] - correct_class_scores[i]
                if score_diff <= -layer_gamma * 2:
                    continue
                
                try:
                    grad_correct = torch.autograd.grad(
                        correct_class_scores[i], layer_activation, 
                        create_graph=True, retain_graph=True, allow_unused=True
                    )[0]
                    
                    if grad_correct is None:
                        continue
                    
                    grad_correct = grad_correct[i]
                    
                    grad_incorrect = torch.autograd.grad(
                        logits[i, confusing_class], layer_activation, 
                        create_graph=True, retain_graph=True, allow_unused=True
                    )[0][i]
                    
                    # Compute gradient difference and norm
                    grad_diff = grad_incorrect - grad_correct
                    grad_norm = self.compute_dual_norm(grad_diff.unsqueeze(0), layer_norm).squeeze() + self.epsilon
                    
                    # Compute margin violation
                    violation = layer_gamma + score_diff / grad_norm
                    violation = F.relu(violation)
                    
                    layer_margin += violation
                    layer_valid_count += 1
                    
                except Exception as e:
                    continue
            
            # Add layer margin if valid
            if layer_valid_count > 0:
                margin_losses.append(layer_margin / layer_valid_count)
        
        # If no valid margin computations, return the fallback - prevents None gradients
        if len(margin_losses) == 0:
            return 0.8 * ce_loss + 0.2 * simple_loss
        
        # avg margin losses across layers
        margin_loss = sum(margin_losses) / len(margin_losses)
        
        ce_weight = max(0.5, 0.9 - self.epoch * 0.05)
        margin_weight = (1.0 - ce_weight) * 0.5
        simple_weight = (1.0 - ce_weight) * 0.5
        
        total_loss = ce_weight * ce_loss + margin_weight * margin_loss + simple_weight * simple_loss
        
        return total_loss