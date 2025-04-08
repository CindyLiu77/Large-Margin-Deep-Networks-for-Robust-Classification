import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CIFARResNet(nn.Module):
    """ResNet model for CIFAR-10 based on Wide ResNet architecture"""
    def __init__(self, block=BasicBlock, num_blocks=[9, 9, 9], num_classes=10, k=10):
        """
        Initialize the ResNet model for CIFAR-10
        
        Args:
            block: Basic building block (BasicBlock by default)
            num_blocks: Number of blocks in each of the 3 stages
            num_classes: Number of output classes (10 for CIFAR-10)
            k: Width factor for the network (wider networks perform better)
        """
        super(CIFARResNet, self).__init__()
        self.in_planes = 16
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Three stages with different feature map sizes (32x32, 16x16, 8x8)
        self.layer1 = self._make_layer(block, 16*k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*k, num_blocks[2], stride=2)
        
        # Final fully connected layer
        self.linear = nn.Linear(64*k, num_classes)
        
        # Store all activations for margin loss
        self.all_activations = []
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_activations=False):
        """
        Forward pass with optional return of intermediate activations
        
        Args:
            x: Input tensor
            return_activations: Whether to return intermediate activations
            
        Returns:
            output logits and optionally a list of activations
        """
        activations = []
        
        # Input
        activations.append(x)
        
        # Initial convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        activations.append(out)  # Conv1 output
        
        # Stage 1
        out = self.layer1(out)
        activations.append(out)  # Layer1 output
        
        # Stage 2
        out = self.layer2(out)
        activations.append(out)  # Layer2 output
        
        # Stage 3
        out = self.layer3(out)
        activations.append(out)  # Layer3 output
        
        # Global average pooling and final classification
        out = F.avg_pool2d(out, 8)
        out_flat = out.view(out.size(0), -1)
        activations.append(out_flat)  # Flattened features
        
        out = self.linear(out_flat)
        activations.append(out)  # Final logits
        
        if return_activations:
            return out, activations
        else:
            return out

def cifar_resnet_small():
    """Small ResNet model for CIFAR-10"""
    return CIFARResNet(BasicBlock, [3, 3, 3], k=2)  # k=2 is a narrower network

def cifar_resnet_medium():
    """Medium ResNet model for CIFAR-10"""
    return CIFARResNet(BasicBlock, [5, 5, 5], k=5)  # k=5 is moderately wide

def cifar_resnet_large():
    """Large ResNet model for CIFAR-10 (close to what's used in the paper)"""
    return CIFARResNet(BasicBlock, [9, 9, 9], k=10)  # Matches the paper's description