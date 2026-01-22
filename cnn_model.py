import torch
import torch.nn as nn
import torch.nn.functional as F

class HabitatCNN(nn.Module):
    def __init__(self, num_classes=71):
        super(HabitatCNN, self).__init__()
        
        # Input shape: (Batch, 15, 35, 35)
        
        # 1. Convolutional Block 1
        # 15 input channels -> 32 filters
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 2. Convolutional Block 2
        # 32 -> 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 3. Convolutional Block 3
        # 64 -> 128 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling to handle any remaining spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 35x35 -> 17x17
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 17x17 -> 8x8
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # Global Pool: 8x8 -> 1x1
        x = self.global_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Dense layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
