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
EPOCHS = 20       # OneCycleLR allows fewer epochs
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()  # Maximize data loading speed
PIN_MEMORY = True if torch.cuda.is_available() else False

class HabitatDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None):
        # Data is already float32. We normalize on the fly or just assume simple scaling.
        # For speed, we'll do simple scaling here if needed.
        # Based on stats (mean ~1241, std ~1208), we standardize.
        self.patches = (patches - 1241.1) / (1208.9 + 1e-6)
        
        # Convert to torch tensor (Channels First: N, 15, 35, 35)
        self.patches = torch.from_numpy(self.patches).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # GPU-friendly Augmentation (doing it here on CPU might be a bottleneck, 
        # but for simple flips it's fine). 
        if self.transform:
            if torch.rand(1) < 0.5:
                patch = torch.flip(patch, [-1]) # Horizontal
            if torch.rand(1) < 0.5:
                patch = torch.flip(patch, [-2]) # Vertical
            # Random 90-degree rotations
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                patch = torch.rot90(patch, k, [-2, -1])
                
        return patch, label

def load_data():
    base_dir = Path(__file__).parent / "data"
    print("Loading data...")
    
    # Try looking in 'train' subdir first (as seen in file structure)
    p1_path = base_dir / "train" / "patches_part1.npy"
    p2_path = base_dir / "train" / "patches_part2.npy"
    csv_path = base_dir / "train.csv"
    
    # Fallback to current directory logic if needed, but structure suggested data/train/
    if not p1_path.exists():
        p1_path = base_dir / "patches_part1.npy"
        p2_path = base_dir / "patches_part2.npy"
    
    part1 = np.load(p1_path)
    part2 = np.load(p2_path)
    patches = np.concatenate([part1, part2], axis=0)
    
    labels_df = pd.read_csv(csv_path)
    # Using the corrected column name
    labels = labels_df["vistgerd_idx"].values
        
    print(f"Data shape: {patches.shape}, Labels: {labels.shape}")
    return patches, labels

def get_model(num_classes=71):
    # Load ResNet18 (lightweight, standard)
    # Weights='DEFAULT' downloads ImageNet weights. 
    # Current torch versions use 'weights=ResNet18_Weights.DEFAULT' instead of pretrained=True
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
    # Strategy: Average the weights of the original 3 channels and replicate/expand
    # Or just kaiming init. Kaiming is safer for training from scratch parts.
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    
    # MODIFY FINAL LAYER
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train():
    print(f"--- STARTING EFFICIENT TRAINING (Device: {DEVICE}) ---")
    
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
    
    # OneCycleLR: The "Superconvergence" scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() # For Mixed Precision
    
    # 3. Training Loop
    best_acc = 0.0
    start_time = time.time()
    
    print(f"Training for {EPOCHS} epochs with Mixed Precision...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # MIXED PRECISION CONTEXT
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Scaled Backward Pass
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
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                # No autocast needed for inference usually, strictly speaking, 
                # but good for consistency. We'll run standard float32 for safety in val.
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
    print(f"DONE! Total time: {total_time:.1f} minutes. Best Acc: {best_acc:.2f}%")
    print("Model saved to model_efficient.pth")

if __name__ == "__main__":
    train()
