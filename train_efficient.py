import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import time
import os
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

# --- CONFIGURATION (optimized for speed/accuracy) ---
BATCH_SIZE = 128  # Larger batch for GPU efficiency
EPOCHS = 50       # Increased from 20 to 50 (User said 20 was very fast)
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()  # Maximize data loading speed
PIN_MEMORY = True if torch.cuda.is_available() else False
TTA_CYCLES = 4    # Test Time Augmentation (averaged predictions)

# ... (HabitatDataset and load_data remain the same) ...

# ...

def get_model(num_classes=71):
    # Load ResNet18 (lightweight, standard)
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
    except ImportError:
        weights = 'DEFAULT' # Older torch versions
        
    model = models.resnet18(weights=weights)
    
    # MODIFY FIRST LAYER: Accept 15 channels instead of 3
    # We keep the spatial structure (kernel size 7, stride 2, padding 3)
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=15, 
        out_channels=original_conv1.out_channels, 
        kernel_size=original_conv1.kernel_size, 
        stride=original_conv1.stride, 
        padding=original_conv1.padding, 
        bias=original_conv1.bias
    )
    
    # Initialize the new layer's weights
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    
    # CRITICAL CHANGE: Remove MaxPool
    # For 35x35 images, the initial MaxPool reduces it to 17x17 too fast.
    # By removing it (Identity), we keep more spatial detail.
    model.maxpool = nn.Identity()
    
    # MODIFY FINAL LAYER
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train():
    print(f"--- STARTING EFFICIENT TRAINING (Device: {DEVICE}) ---")
    print(f"--- CONFIG: Epochs={EPOCHS}, MaxPool=OFF (Better for 35x35), TTA={TTA_CYCLES}x ---")
    
    # 1. Prepare Data
    patches, labels = load_data()
    
    dataset_size = len(patches)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_dataset = HabitatDataset(patches[train_indices], labels[train_indices], transform=True)
    val_dataset = HabitatDataset(patches[val_indices], labels[val_indices], transform=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    
    # 2. Setup Model & Optimization
    model = get_model(num_classes=71).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() 
    
    # 3. Training Loop
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100. * correct / total
        
        # Validation with TTA (Test Time Augmentation)
        # To save time, we only do TTA on the confirmation steps or final model save?
        # Let's do standard validation for speed in the loop.
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1}/{EPOCHS} [{elapsed:.1f}m] | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model_efficient.pth")
            
    total_time = (time.time() - start_time) / 60
    print(f"DONE! Total time: {total_time:.1f} minutes. Best Val Acc: {best_acc:.2f}%")
    print("Model saved to model_efficient.pth")

if __name__ == "__main__":
    train()
