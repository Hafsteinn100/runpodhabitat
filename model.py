import joblib
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PTH = BASE_DIR / "model_efficient.pth"  # PyTorch (Best)
MODEL_JOBLIB = BASE_DIR / "model.joblib"      # Sklearn (Fast Baseline)

model = None
model_type = None


# --- CUSTOM ARCHITECTURE (Must match train_efficient.py) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class SmallResNet(nn.Module):
    def __init__(self, num_classes=71):
        super(SmallResNet, self).__init__()
        self.conv1 = nn.Conv2d(15, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# --- LOADER LOGIC ---
try:
    if MODEL_PTH.exists():
        print(f"Loading PyTorch Model from {MODEL_PTH}...")
        
        # Load Custom SmallResNet
        p_model = SmallResNet(num_classes=71)
        
        # Load Weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p_model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
        p_model.to(device)
        p_model.eval()
        
        model = p_model
        model_type = "pytorch"
        print("SUCCESS: PyTorch Model Loaded.")
        
    elif MODEL_JOBLIB.exists():
        print(f"Loading Sklearn Model from {MODEL_JOBLIB}...")
        model = joblib.load(MODEL_JOBLIB)
        model_type = "sklearn"
        print("SUCCESS: Sklearn Model Loaded.")
        
    else:
        print("ERROR: No model file found! (Checked model_efficient.pth and model.joblib)")

        
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")

def predict(patch):
    """
    Hybrid Predictor: Handles both 1D Flattened (Sklearn) and 3D Tensor (PyTorch) inputs.
    Patch shape: (15, 35, 35) numpy array
    """
    if model is None:
        raise RuntimeError("Model is not loaded! Run a training script first.")

    if model_type == "pytorch":
        # Preprocess for PyTorch: (15, 35, 35) -> (1, 15, 35, 35) Tensor
        # Also Normalize (matching train_efficient.py logic)
        patch_norm = (patch - 1241.1) / (1208.9 + 1e-6)
        
        # Convert to Tensor (on CPU first)
        tensor = torch.from_numpy(patch_norm).float()
        
        device = next(model.parameters()).device
        
        with torch.no_grad():
            # TTA (Test Time Augmentation) - 4x Strategy
            # 1. Original
            inputs = [tensor] 
            # 2. Horizontal Flip
            inputs.append(torch.flip(tensor, [-1]))
            # 3. Vertical Flip
            inputs.append(torch.flip(tensor, [-2]))
            # 4. Rot90
            inputs.append(torch.rot90(tensor, 1, [-2, -1]))
            
            # Batch them: (4, 15, 35, 35)
            batch = torch.stack(inputs).to(device)
            
            outputs = model(batch)
            # Average the logits or probabilities
            # Simple averaging of logits is usually fine
            avg_output = outputs.mean(dim=0, keepdim=True)
            
            _, prediction = avg_output.max(1)
            return int(prediction.item())
            
    elif model_type == "sklearn":
        # Preprocess for Sklearn: Flatten -> (1, 18375)
        flat = patch.reshape(1, -1)
        prediction = model.predict(flat)
        return int(prediction[0])
        
    return 0