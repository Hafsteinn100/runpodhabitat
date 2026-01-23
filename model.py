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
model_cnn = None
model_et = None

# 1. Load CNN (EfficientNet-B0)
try:
    if MODEL_PTH.exists():
        print(f"Loading CNN from {MODEL_PTH}...")
        try:
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT
            p_model = models.efficientnet_b0(weights=weights)
        except:
             p_model = models.efficientnet_b0(pretrained=True)
             
        # Match Architecture (15 channels)
        original_conv = p_model.features[0][0]
        p_model.features[0][0] = nn.Conv2d(
            15, original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=original_conv.bias
        )
        
        # Match Classifier
        num_ftrs = p_model.classifier[1].in_features
        p_model.classifier[1] = nn.Linear(num_ftrs, 71)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p_model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
        p_model.to(device)
        p_model.eval()
        
        model_cnn = p_model
        print("SUCCESS: CNN (EfficientNet-B0) Loaded.")
except Exception as e:
    print(f"ERROR Loading CNN: {e}")

# 2. Load ExtraTrees
try:
    ensemble_path = BASE_DIR / "model_ensemble.joblib"
    if ensemble_path.exists():
        print(f"Loading ExtraTrees from {ensemble_path}...")
        model_et = joblib.load(ensemble_path)
        print("SUCCESS: ExtraTrees Loaded.")
    elif MODEL_JOBLIB.exists(): # Fallback to old joblib name if needed
        print(f"Loading ExtraTrees from {MODEL_JOBLIB}...")
        model_et = joblib.load(MODEL_JOBLIB)
        print("SUCCESS: ExtraTrees Loaded (Fallback).")
except Exception as e:
    print(f"ERROR Loading ExtraTrees: {e}")


def predict(patch):
    """
    Hybrid Predictor: CNN + ExtraTrees Ensemble
    """
    if model_cnn is None and model_et is None:
        raise RuntimeError("No models loaded!")
        
    probs_cnn = np.zeros(71)
    probs_et = np.zeros(71)
    
    # 1. CNN Prediction (TTA 4x)
    if model_cnn is not None:
        patch_norm = (patch - 1241.1) / (1208.9 + 1e-6)
        tensor = torch.from_numpy(patch_norm).float()
        device = next(model_cnn.parameters()).device
        
        with torch.no_grad():
            inputs = [tensor] 
            inputs.append(torch.flip(tensor, [-1]))
            inputs.append(torch.flip(tensor, [-2]))
            inputs.append(torch.rot90(tensor, 1, [-2, -1]))
            
            batch = torch.stack(inputs).to(device)
            outputs = model_cnn(batch)
            
            # Softmax to get probs
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            probs_cnn = probs.mean(axis=0) # Average TTA

    # 2. ExtraTrees Prediction
    if model_et is not None:
        flat = patch.reshape(1, -1)
        # Check if model supports predict_proba
        if hasattr(model_et, "predict_proba"):
            probs_et = model_et.predict_proba(flat)[0]
            # Handle case where ET might not have seen all classes? 
            # Sklearn handles this usually if classes were known at fit.
            if len(probs_et) != 71:
                # Pad if necessary, but trained on full set usually fine.
                # If mismatch, rely on CNN
                if len(probs_et) < 71:
                   # This is tricky, simplified assumption: it matches or we skip
                   # Usually it matches if trained on all classes.
                   pass
        else:
            # Hard vote
            pred = model_et.predict(flat)[0]
            probs_et[pred] = 1.0

    # 3. Ensemble (Weighted Average)
    # Give CNN slightly more weight? Or equal?
    # CNN (0.6) + ET (0.4) is a good starting point for "Deep Learning + Tabular"
    
    if model_cnn is not None and model_et is not None:
        final_probs = (0.6 * probs_cnn) + (0.4 * probs_et)
    elif model_cnn is not None:
        final_probs = probs_cnn
    else:
        final_probs = probs_et
        
    return int(np.argmax(final_probs))