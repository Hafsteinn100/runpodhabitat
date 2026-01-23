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

# --- LOADER LOGIC ---
try:
    if MODEL_PTH.exists():
        print(f"Loading PyTorch Model from {MODEL_PTH}...")
        
        # Re-define architecture (Must match train_efficient.py)
        # ResNet34 with 15 channels
        try:
            from torchvision.models import ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT
        except:
             weights = 'DEFAULT'
             
        from torchvision.models import resnet34
        p_model = resnet34(weights=weights)
        original_conv1 = p_model.conv1
        p_model.conv1 = nn.Conv2d(
            in_channels=15, 
            out_channels=original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias
        )
        
        # MaxPool is RESTORED (We used Identity before, now using standard)
        # p_model.maxpool remains standard.
        
        p_model.fc = nn.Linear(p_model.fc.in_features, 71)
        
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
        
        tensor = torch.from_numpy(patch_norm).float().unsqueeze(0)
        
        device = next(model.parameters()).device
        tensor = tensor.to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            _, prediction = outputs.max(1)
            return int(prediction.item())
            
    elif model_type == "sklearn":
        # Preprocess for Sklearn: Flatten -> (1, 18375)
        flat = patch.reshape(1, -1)
        prediction = model.predict(flat)
        return int(prediction[0])
        
    return 0