import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from cnn_model import HabitatCNN
import copy

import json
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.functional as TF

# Config
BATCH_SIZE = 64
EPOCHS = 100 
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HabitatDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None, mean=None, std=None):
        # Normalize: (x - mean) / std
        if mean is not None and std is not None:
             patches = (patches - mean) / (std + 1e-6)
             
        self.patches = torch.from_numpy(patches).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # Augmentation
        if self.transform:
            # Flips
            if torch.rand(1) < 0.5:
                patch = torch.flip(patch, [-1])
            if torch.rand(1) < 0.5:
                patch = torch.flip(patch, [-2])
            
            # Rotation (90, 180, 270)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                patch = torch.rot90(patch, k, [-2, -1])
                
        return patch, label

def load_data():
    base_dir = Path(__file__).parent / "data"
    
    print("Loading data...")
    try:
        part1 = np.load(base_dir / "train" / "patches_part1.npy")
        part2 = np.load(base_dir / "train" / "patches_part2.npy")
        patches = np.concatenate([part1, part2], axis=0)
        
        labels_df = pd.read_csv(base_dir / "train.csv")
        labels = labels_df["vistgerd_idx"].values
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return dummy data for testing if files missing
        return np.random.randn(100, 15, 35, 35).astype(np.float32), np.random.randint(0, 71, 100)
    
    print(f"Data shape: {patches.shape}, Labels: {labels.shape}")
    return patches, labels

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    patches, labels = load_data()
    
    # Calculate Normalization Stats
    print("Calculating normalization stats...")
    mean = np.mean(patches)
    std = np.std(patches)
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Save stats
    stats = {"mean": float(mean), "std": float(std)}
    with open(Path(__file__).parent / "normalization.json", "w") as f:
        json.dump(stats, f)
    
    # Split
    dataset_size = len(patches)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create datasets
    train_dataset = HabitatDataset(patches[train_indices], labels[train_indices], transform=True, mean=mean, std=std)
    val_dataset = HabitatDataset(patches[val_indices], labels[val_indices], transform=False, mean=mean, std=std)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Setup Model
    model = HabitatCNN(num_classes=71).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 3. Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
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
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # print(f"  New best model: {best_acc:.2f}%")
            
    print(f"Training complete. Best Validation Accuracy: {best_acc:.2f}%")
    
    # 4. Save best model
    model.load_state_dict(best_model_wts)
    save_path = Path(__file__).parent / "model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
