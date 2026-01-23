import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.models as models
import os

# --- CONFIGURATION ---
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess():
    print("Loading Data...")
    # Load Train
    X_part1 = np.load("data/patches_part1.npy")
    X_part2 = np.load("data/patches_part2.npy")
    X_train = np.concatenate([X_part1, X_part2], axis=0)
    
    df_train = pd.read_csv("data/train.csv")
    y_train_raw = df_train['vistgerd_idx'].values
    
    # Load Test
    X_test = None
    if os.path.exists("data/test/patches_test.npy"):
        X_test = np.load("data/test/patches_test.npy")
    elif os.path.exists("data/patches_test.npy"):
        X_test = np.load("data/patches_test.npy")
    else:
        print("WARNING: Test data not found. Predictions will be skipped.")
    
    # --- NORMALIZATION (Restored to 80% config) ---
    print("Normalizing data (Standardization)...")
    # Using stats from successful run: Mean ~1241, Std ~1208
    X_train = (X_train - 1241.1) / (1208.9 + 1e-6)
    
    if X_test is not None:
         X_test = (X_test - 1241.1) / (1208.9 + 1e-6)

    # Convert to Float32/Channels First logic if needed?
    # Data seems to be (N, 15, 35, 35). Torch handles float32.
    X_train = X_train.astype(np.float32)
    if X_test is not None:
        X_test = X_test.astype(np.float32)

    # Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    
    return X_train, y_train_enc, X_test, le

def get_efficientnet(num_classes):
    # UPGRADE: EfficientNet-B0
    try:
        from torchvision.models import EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except:
        model = models.efficientnet_b0(pretrained=True)
    
    # Adapt first layer to 15 channels
    original_conv = model.features[0][0]
    
    # Smart Init
    with torch.no_grad():
        old_weights = original_conv.weight
        avg_weight = torch.mean(old_weights, dim=1, keepdim=True)
        new_weights = avg_weight.repeat(1, 15, 1, 1)
        
    model.features[0][0] = nn.Conv2d(
        15, original_conv.out_channels, 
        kernel_size=original_conv.kernel_size, 
        stride=original_conv.stride, 
        padding=original_conv.padding, 
        bias=original_conv.bias
    )
    model.features[0][0].weight.data = new_weights
    
    # Adapt Classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def train_model():
    X, y, X_test, le = load_and_preprocess()
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Datasets
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model
    model = get_efficientnet(num_classes=len(le.classes_)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    # OneCycleLR is great for super convergence
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"Starting Training EfficientNet-B0 on {DEVICE}...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Basic Augmentation (Flip/Rotate)
            if np.random.rand() < 0.5:
                images = torch.flip(images, [-1])
            if np.random.rand() < 0.5:
                images = torch.flip(images, [-2])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
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

    # --- PREDICTION ---
    if X_test is not None:
        print("\nGenerating Test Predictions with TTA...")
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
        model.eval()
        
        X_test_tensor = torch.tensor(X_test).to(DEVICE)
        
        with torch.no_grad():
            # TTA: Original + Flips
            out1 = torch.softmax(model(X_test_tensor), dim=1).cpu().numpy()
            out2 = torch.softmax(model(torch.flip(X_test_tensor, [3])), dim=1).cpu().numpy()
            out3 = torch.softmax(model(torch.flip(X_test_tensor, [2])), dim=1).cpu().numpy()

        final_probs = (out1 + out2 + out3) / 3.0
        np.save("probs_efficient.npy", final_probs)
        print("Saved 'probs_efficient.npy'.")
    else:
        print("\nSkipping Prediction (No Test Data).")

if __name__ == "__main__":
    train_model()