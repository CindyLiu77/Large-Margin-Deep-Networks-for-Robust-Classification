import torch
import torch.nn as nn
import torch.nn.functional as F

class LargeMarginLoss(nn.Module):
    def __init__(self, gamma=1.0, norm="l2", aggregation="max", epsilon=1e-6):
        '''
        gamma: margin param
        norm: norm type ('l1', 'l2', 'linf')
        aggregation: aggregation type for incorrect classes ('max' or 'sum')
        epsilon: small value to avoid division by zero
        '''
        super(LargeMarginLoss, self).__init__()
        self.gamma = gamma
        self.norm = norm
        self.aggregation = aggregation
        self.epsilon = epsilon

    def forward(self, x, y):
        '''
        x: input tensor (batch_size, num_classes)
        y: target tensor (batch_size, num_classes)
        '''
        
        batch_size, num_classes = x.size()
        correct_class_scores = x[torch.arange(batch_size), y].unsqueeze(1)

        diff = x - correct_class_scores
        diff.scatter_(1, y.unsqueeze(1), float('-inf'))

        if self.aggregation == 'max':
            margin_loss = torch.max(self.gamma + diff, dim=1)[0]
        elif self.aggregation == 'sum':
            margin_loss = torch.sum(torch.exp(self.gamma + diff), dim=1)
        else:
            raise ValueError("Invalid aggregation type")
        
        return torch.mean(margin_loss)

