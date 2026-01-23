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
BATCH_SIZE = 128
EPOCHS = 100      # Optimized: 100 epochs for full convergence (still fast)
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True if torch.cuda.is_available() else False
TTA_CYCLES = 4

class HabitatDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None):
        self.patches = (patches - 1241.1) / (1208.9 + 1e-6)
        
        self.patches = torch.from_numpy(self.patches).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Standard Augmentation (Tried & True for high accuracy)
            if torch.rand(1) < 0.5:
                patch = torch.flip(patch, [-1])
            if torch.rand(1) < 0.5:
                patch = torch.flip(patch, [-2])
            
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                patch = torch.rot90(patch, k, [-2, -1])
            
        return patch, label

def load_data():
    base_dir = Path(__file__).parent / "data"
    print("Loading data...")
    
    possible_roots = ["data/train", "data", "."]
    p1_path = None
    p2_path = None
    csv_path = None
    
    for root in possible_roots:
        p1 = base_dir / root / "patches_part1.npy"
        if not p1.exists():
            p1 = Path(root) / "patches_part1.npy"
        if p1.exists():
            p1_path = p1
            p2_path = p1.parent / "patches_part2.npy"
            print(f"Found data in: {p1.parent}")
            break
            
    if p1_path is None:
        if os.path.exists("data/train/patches_part1.npy"):
             p1_path = "data/train/patches_part1.npy"
             p2_path = "data/train/patches_part2.npy"
        elif os.path.exists("patches_part1.npy"):
             p1_path = "patches_part1.npy"
             p2_path = "patches_part2.npy"
        else:
             raise FileNotFoundError("Could not find patches_part1.npy")

    # Finalize paths as Path objects or strings
    part1 = np.load(str(p1_path))
    part2 = np.load(str(p2_path))
    
    for root in possible_roots:
        c = base_dir / root / "train.csv"
        if not c.exists():
            c = Path(root) / "train.csv"
        if c.exists():
            csv_path = c
            break
            
    if csv_path is None:
         if os.path.exists("train.csv"):
             csv_path = "train.csv"
         else:
             raise FileNotFoundError("Could not find train.csv")
    
    part1 = np.load(p1_path)
    part2 = np.load(p2_path)
    patches = np.concatenate([part1, part2], axis=0)
    
    labels_df = pd.read_csv(csv_path)
    labels = labels_df["vistgerd_idx"].values
        
    print(f"Data shape: {patches.shape}, Labels: {labels.shape}")
    return patches, labels

def get_model(num_classes=71):
    # BACK TO RESNET18 (Best performance so far: 80%)
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
    except ImportError:
        weights = 'DEFAULT'
        
    model = models.resnet18(weights=weights)
    
    # MODIFY FIRST LAYER: Accept 15 channels
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=15, 
        out_channels=original_conv1.out_channels, 
        kernel_size=original_conv1.kernel_size, 
        stride=original_conv1.stride, 
        padding=original_conv1.padding, 
        bias=original_conv1.bias
    )
    
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    
    # Standard MaxPool (Performance critical)
    
    # MODIFY FINAL LAYER
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train():
    print(f"--- STARTING EFFICIENT TRAINING (Device: {DEVICE}) ---")
    print(f"--- CONFIG: ResNet18 (Best), Epochs={EPOCHS}, MaxPool=ON ---")
    
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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    
    model = get_model(num_classes=71).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    
    # Standard CrossEntropy (No LabelSmoothing - aiming for sharp fit)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() 
    
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
