import torch
import numpy as np
from pathlib import Path
from model import predict

def verify():
    print("Verifying model prediction...")
    
    # Create a dummy patch
    dummy = np.random.randn(15, 35, 35).astype(np.float32)
    
    try:
        cls = predict(dummy)
        print(f"Prediction for dummy input: {cls}")
    except Exception as e:
        print(f"FAILED to predict: {e}")
        return

    # Check accuracy on a few real samples
    print("Checking accuracy on subset...")
    base_dir = Path(__file__).parent / "data"
    try:
        part1 = np.load(base_dir / "train" / "patches_part1.npy")
        labels = pd.read_csv(base_dir / "train.csv")["vistgerd_idx"].values
        
        correct = 0
        total = 100
        for i in range(total):
            p = part1[i]
            true_label = labels[i]
            pred = predict(p)
            if pred == true_label:
                correct += 1
                
        print(f"Subset Accuracy (Train Data): {correct}/{total} = {correct}%")
    except Exception as e:
        print(f"Skipping subset check: {e}")

if __name__ == "__main__":
    import pandas as pd # Needed inside try block
    verify()
