import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_activations=False):
        activations = []
        
        # input
        activations.append(x)
        
        # 1
        x = self.conv1(x)
        activations.append(x)
        x = F.relu(x)
        
        # 2
        x = self.conv2(x)
        activations.append(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # 3
        x = torch.flatten(x, 1)
        activations.append(x)
        
        # 4
        x = self.fc1(x)
        activations.append(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # 5
        x = self.fc2(x)
        activations.append(x)
        
        if return_activations:
            return x, activations
        else:
            return x