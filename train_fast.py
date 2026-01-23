import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

def train_rf():
    print("--- STARTING RANDOM FOREST TRAINING ---")
    
    # 1. Load Train
    X_part1 = np.load("data/train/patches_part1.npy")
    X_part2 = np.load("data/train/patches_part2.npy")
    X = np.concatenate([X_part1, X_part2], axis=0)
    
    df_train = pd.read_csv("data/train/train.csv")
    y = df_train['label'].values
    
    # Flatten for RF
    X_flat = X.reshape(X.shape[0], -1)

    # Encode Labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 2. Load Test
    X_test = np.load("data/test/patches_test.npy")
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 3. Train
    print("Training Random Forest (200 trees)...")
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_flat, y_enc)
    
    print("Training complete. Starting TTA Prediction...")

    # 4. Predict with TTA (Flip Augmentation)
    # Original
    probs_orig = clf.predict_proba(X_test_flat)
    
    # Flipped (We have to reshape to 3D, flip, then flatten again)
    X_test_flipped = np.flip(X_test, axis=2) # Flip width
    X_test_flipped_flat = X_test_flipped.reshape(X_test.shape[0], -1)
    probs_flip = clf.predict_proba(X_test_flipped_flat)
    
    # Average
    final_probs = (probs_orig + probs_flip) / 2.0
    
    # Save Probabilities
    np.save("probs_rf.npy", final_probs)
    
    # Save Classes (needed for the ensemble script to know the names)
    np.save("classes.npy", le.classes_)
    
    print("Saved 'probs_rf.npy' and 'classes.npy' for ensemble.")

if __name__ == "__main__":
    train_rf()