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
        
        # ce for stability
        ce_loss = self.cross_entropy(logits, targets)
        
        targets_one_hot = F.one_hot(targets, num_classes).float()
        correct_class_scores = torch.sum(logits * targets_one_hot, dim=1, keepdim=True)
        
        # Gamma + (incorrect_score - correct_score)
        margin_violations = self.gamma + logits - correct_class_scores
        mask = 1.0 - targets_one_hot
        margin_violations = margin_violations * mask
        margin_violations = F.relu(margin_violations)
        
        if self.aggregation == "max":
            margin_loss = torch.max(margin_violations, dim=1)[0]
        else:
            margin_loss = torch.sum(margin_violations, dim=1)
        
        margin_loss = torch.mean(margin_loss)
        
        # weighted loss doesnt follow the paper but stable
        total_loss = 0.8 * ce_loss + 0.2 * margin_loss
        
        return total_loss


class LargeMarginLoss(nn.Module):
    """
    Single Layer Large Margin Loss
    """
    def __init__(self, gamma=1.0, norm="l2", aggregation="max", epsilon=1e-6):
        super(LargeMarginLoss, self).__init__()
        self.gamma = gamma
        self.norm = norm
        self.aggregation = aggregation
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss()
        # Use simple margin loss as a fallback for None Grad
        self.simple_margin = SimpleLargeMarginLoss(gamma=gamma, aggregation=aggregation)
    
    def compute_dual_norm(self, grad_diff):
        """
        Compute the dual norm of gradient differences
        """
        if self.norm == "l1":
            return torch.amax(torch.abs(grad_diff), dim=list(range(1, grad_diff.dim())))
        elif self.norm == "l2":
            return torch.sqrt(torch.sum(grad_diff**2, dim=list(range(1, grad_diff.dim()))) + self.epsilon)
        elif self.norm == "linf":
            return torch.sum(torch.abs(grad_diff), dim=list(range(1, grad_diff.dim())))
        else:
            raise ValueError(f"Unsupported norm type: {self.norm}")
    
    def forward(self, logits, targets, layer_activations=None):
        """
        Forward pass of the large margin loss
        """
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
    multi-layer margin loss implementation with performance changes
    - Uses weighted cross-entropy  and simple margin loss as fallback for None gradients
    - Uses a subset of the batch for margin computation
    - Uses adaptive weighting for margin loss based on epoch
    - only most confusing class is used for margin computation
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
        
        # start with a small subset of the batch for efficiency - increases with epochs
        effective_batch_size = min(batch_size, 4 + self.epoch * 2)
        sample_indices = torch.randperm(batch_size)[:effective_batch_size]
        
        for layer_idx in active_layers:
            layer_gamma = self.gamma[self.layers.index(layer_idx)]
            layer_norm = self.norm[self.layers.index(layer_idx)]
            
            layer_activation = activations[layer_idx]
            if layer_activation is None:
                continue
                
            if not layer_activation.requires_grad:
                layer_activation.requires_grad_(True)
            
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
                    
                    grad_diff = grad_incorrect - grad_correct
                    grad_norm = self.compute_dual_norm(grad_diff.unsqueeze(0), layer_norm).squeeze() + self.epsilon
                    
                    violation = layer_gamma + score_diff / grad_norm
                    violation = F.relu(violation)
                    
                    layer_margin += violation
                    layer_valid_count += 1
                    
                except Exception as e:
                    continue
            
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
    
class TrueMultiLayerMarginLoss(nn.Module):
    """
    Strict implementation of the Large Margin Deep Networks paper.
    - Applies margin to all layers simultaneously
    - Flexible choice of incorrect classes (all or top-k)
    - No cross-entropy mixing
    - Uses only the gradient-based margin formulation
    - Supports both vectorized and sample-by-sample computation
    - vectorized version is faster but less robust
    """
    def __init__(self, layers, gamma=1.0, norm="l2", aggregation="max", epsilon=1e-6, vectorize=False, top_k=None):
        super(TrueMultiLayerMarginLoss, self).__init__()
        self.layers = layers
        self.vectorize = vectorize
        self.top_k = top_k  # If None or -1, consider all incorrect classes
        
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
        
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def compute_dual_norm(self, grad_diff, norm_type):
        """
        Compute the dual norm of gradient differences
        """
        if norm_type == "l1":
            return torch.amax(torch.abs(grad_diff), dim=list(range(1, grad_diff.dim())))
        elif norm_type == "l2":
            return torch.sqrt(torch.sum(grad_diff**2, dim=list(range(1, grad_diff.dim()))) + self.epsilon)
        elif norm_type == "linf":
            return torch.sum(torch.abs(grad_diff), dim=list(range(1, grad_diff.dim())))
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
    
    def forward_vectorized(self, logits, targets, activations):
        """
        Vectorized implementation for efficiency
        """
        batch_size, num_classes = logits.size()
        device = logits.device
        
        targets_one_hot = F.one_hot(targets, num_classes).float().to(device)
        
        correct_class_scores = torch.sum(logits * targets_one_hot, dim=1)
        
        total_loss = torch.tensor(0.0, device=device)
        valid_loss_count = 0
        
        for layer_idx, gamma in zip(self.layers, self.gamma):
            norm_type = self.norm[self.layers.index(layer_idx)]
            layer_activation = activations[layer_idx]
            
            if layer_activation is None:
                continue
                
            if not layer_activation.requires_grad:
                layer_activation.requires_grad_(True)
            
            layer_loss = torch.tensor(0.0, device=device)
            layer_valid_count = 0
            
            try:
                # Vectorized gradient computation for correct class scores
                batch_correct_grads = torch.autograd.grad(
                    outputs=correct_class_scores,
                    inputs=layer_activation,
                    grad_outputs=torch.ones_like(correct_class_scores),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if batch_correct_grads is None:
                    continue
                
                for i in range(batch_size):
                    target = targets[i]
                    correct_score = correct_class_scores[i]
                    grad_correct = batch_correct_grads[i]
                    
                    # top k incorrect classes
                    logits_without_correct = logits[i].clone()
                    logits_without_correct[target] = -float('inf')
                    
                    if self.top_k is None or self.top_k <= 0:
                        confusing_classes = torch.nonzero(torch.ones_like(logits_without_correct).to(device) * (torch.arange(num_classes).to(device) != target)).squeeze()
                    else:
                        confusing_classes = torch.argsort(logits_without_correct, descending=True)[:self.top_k]
                    
                    sample_violations = []
                    for j in confusing_classes:
                        j = j.item()
                        try:
                            # grad_correct = torch.autograd.grad(
                            #     correct_score, layer_activation, 
                            #     create_graph=True, retain_graph=True, allow_unused=True
                            # )[0]
                            
                            # if grad_correct is None:
                            #     continue
                            
                            # grad_correct = grad_correct[i]
                            
                            # Can't vectorize this part due to different incorrect classes
                            grad_incorrect = torch.autograd.grad(
                                logits[i, j], 
                                layer_activation, 
                                create_graph=True, 
                                retain_graph=True, 
                                allow_unused=True
                            )[0]
                            
                            if grad_incorrect is None:
                                continue
                                
                            grad_incorrect = grad_incorrect[i]
                            grad_diff = grad_incorrect - grad_correct
                            
                            # grad_norm = self.compute_dual_norm(grad_diff.unsqueeze(0), norm_type).squeeze() + self.epsilon

                            #  first order taylor approx instead of computation of second order derivative with .detach()
                            grad_norm = self.compute_dual_norm(grad_diff.unsqueeze(0), norm_type).squeeze() + self.epsilon
                            grad_norm = grad_norm.detach()
                            
                            score_diff = logits[i, j] - correct_score
                            violation = gamma + score_diff / grad_norm
                            violation = F.relu(violation)
                            
                            sample_violations.append(violation)
                        except Exception as e:
                            continue
                    
                    if sample_violations:
                        if self.aggregation == "max":
                            sample_loss = torch.max(torch.stack(sample_violations))
                        else:
                            sample_loss = torch.sum(torch.stack(sample_violations))
                        
                        layer_loss += sample_loss
                        layer_valid_count += 1
                
                if layer_valid_count > 0:
                    layer_loss = layer_loss / layer_valid_count
                    total_loss += layer_loss
                    valid_loss_count += 1
                    
            except Exception as e:
                continue
        
        # Fallback
        if valid_loss_count == 0:
            return self.cross_entropy(logits, targets)
        
        return total_loss / valid_loss_count
    
    def forward_sample_by_sample(self, logits, targets, activations):
        """
        Sample-by-sample implementation for robustness
        """
        batch_size, num_classes = logits.size()
        device = logits.device
        
        targets_one_hot = F.one_hot(targets, num_classes).float().to(device)
        
        correct_class_scores = torch.sum(logits * targets_one_hot, dim=1)

        total_loss = torch.tensor(0.0, device=device)
        valid_loss_count = 0
        
        for layer_idx, gamma in zip(self.layers, self.gamma):
            norm_type = self.norm[self.layers.index(layer_idx)]
            layer_activation = activations[layer_idx]
            
            if layer_activation is None:
                continue
                
            if not layer_activation.requires_grad:
                layer_activation.requires_grad_(True)
            
            layer_loss = torch.tensor(0.0, device=device)
            layer_valid_count = 0
            
            # Process samples one by one
            for i in range(batch_size):
                target = targets[i]
                correct_score = correct_class_scores[i]
                
                try:
                    # gradient for correct class
                    grad_correct = torch.autograd.grad(
                        correct_score, layer_activation, 
                        create_graph=True, retain_graph=True, allow_unused=True
                    )[0]
                    
                    if grad_correct is None:
                        continue
                    
                    grad_correct = grad_correct[i]
                    
                    # top k incorrect classes
                    logits_without_correct = logits[i].clone()
                    logits_without_correct[target] = -float('inf')
                    
                    if self.top_k is None or self.top_k <= 0:
                        confusing_classes = torch.nonzero(torch.ones_like(logits_without_correct).to(device) * (torch.arange(num_classes).to(device) != target)).squeeze()
                    else:
                        confusing_classes = torch.argsort(logits_without_correct, descending=True)[:self.top_k]
                    
                    sample_violations = []
                    for j in confusing_classes:
                        j = j.item()
                        try:
                            grad_incorrect = torch.autograd.grad(
                                logits[i, j], 
                                layer_activation, 
                                create_graph=True, 
                                retain_graph=True, 
                                allow_unused=True
                            )[0]
                            
                            if grad_incorrect is None:
                                continue
                                
                            grad_incorrect = grad_incorrect[i]
                            
                            grad_diff = grad_incorrect - grad_correct
                            grad_norm = self.compute_dual_norm(grad_diff.unsqueeze(0), norm_type).squeeze() + self.epsilon
                            
                            #  first order taylor approx instead of computation of second order derivative with .detach()
                            grad_norm = grad_norm.detach()
                            
                            score_diff = logits[i, j] - correct_score
                            
                            violation = gamma + score_diff / grad_norm
                            violation = F.relu(violation)
                            
                            sample_violations.append(violation)
                            
                        except Exception as e:
                            continue
                            
                    if sample_violations:
                        if self.aggregation == "max":
                            sample_loss = torch.max(torch.stack(sample_violations))
                        else:
                            sample_loss = torch.sum(torch.stack(sample_violations))
                        
                        layer_loss += sample_loss
                        layer_valid_count += 1
                        
                except Exception as e:
                    continue
            
            if layer_valid_count > 0:
                layer_loss = layer_loss / layer_valid_count
                total_loss += layer_loss
                valid_loss_count += 1
        
        # Fallback
        if valid_loss_count == 0:
            return self.cross_entropy(logits, targets)
        
        return total_loss / valid_loss_count
    
    def forward(self, logits, targets, activations=None):
        """
        Compute the multi-layer margin loss exactly like in paper
        """
        # Fallback
        if activations is None or not self.training:
            return self.cross_entropy(logits, targets)
        
        if len(activations) <= max(self.layers):
            raise ValueError(f"Not enough activations ({len(activations)}) for requested layers (max {max(self.layers)})")
        
        if self.vectorize:
            return self.forward_vectorized(logits, targets, activations)
        else:
            return self.forward_sample_by_sample(logits, targets, activations)