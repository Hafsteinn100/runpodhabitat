import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION (Custom CNN for 15 Channels) ---
class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SatelliteCNN, self).__init__()
        
        # Input: 15 channels, 35x35 image
        self.features = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 17x17
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 8x8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # -> 1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_and_preprocess():
    print("Loading Data...")
    # Load Train
    X_part1 = np.load("data/train/patches_part1.npy")
    X_part2 = np.load("data/train/patches_part2.npy")
    X_train = np.concatenate([X_part1, X_part2], axis=0)
    
    # Load Labels
    df_train = pd.read_csv("data/train/train.csv")
    y_train_raw = df_train['label'].values
    
    # Load Test
    X_test = np.load("data/test/patches_test.npy")
    df_test = pd.read_csv("data/test/test.csv")
    
    # --- CRITICAL: NORMALIZATION ---
    # Sentinel data is 12-bit (0-4096+). Neural nets need 0-1.
    print("Normalizing data (Dividing by 10000)...")
    X_train = np.clip(X_train.astype(np.float32) / 10000.0, 0, 1)
    X_test = np.clip(X_test.astype(np.float32) / 10000.0, 0, 1)

    # Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    
    return X_train, y_train_enc, X_test, df_test, le

def train_model():
    X, y, X_test, df_test, le = load_and_preprocess()
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Convert to Tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize
    model = SatelliteCNN(num_classes=len(le.classes_)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Training on {DEVICE}...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Val Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")

    # --- PREDICTION WITH TTA (Test Time Augmentation) ---
    print("\nGenerating Test Predictions with TTA...")
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE)) # Load best weights
    model.eval()
    
    X_test_tensor = torch.tensor(X_test).to(DEVICE)
    
    with torch.no_grad():
        # 1. Original
        outputs_orig = torch.softmax(model(X_test_tensor), dim=1).cpu().numpy()
        
        # 2. Horizontal Flip
        X_flip = torch.flip(X_test_tensor, [3]) # Flip width dim
        outputs_flip = torch.softmax(model(X_flip), dim=1).cpu().numpy()
        
        # 3. Vertical Flip
        X_vflip = torch.flip(X_test_tensor, [2]) # Flip height dim
        outputs_vflip = torch.softmax(model(X_vflip), dim=1).cpu().numpy()

    # Average the probabilities
    final_probs = (outputs_orig + outputs_flip + outputs_vflip) / 3.0
    
    # Save Probabilities for Ensemble
    np.save("probs_efficient.npy", final_probs)
    print("Saved 'probs_efficient.npy' for ensemble.")

if __name__ == "__main__":
    train_model()