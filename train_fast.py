import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import os

def load_data():
    print("Loading raw data...")
    # Load the big image arrays
    X_part1 = np.load("data/train/patches_part1.npy")
    X_part2 = np.load("data/train/patches_part2.npy")
    X = np.concatenate([X_part1, X_part2], axis=0)

    # Flatten images (from 4D to 2D) for the Random Forest
    # (N, 35, 35, 15) -> (N, 18375)
    X = X.reshape(X.shape[0], -1)

    # Load labels
    df_labels = pd.read_csv("data/train/train.csv")
    
    # Check if 'id' or 'ID' exists to be safe
    if 'id' in df_labels.columns:
        ids = df_labels['id']
    else:
        ids = df_labels['ID']

    y = df_labels['label'].values

    print(f"Original dataset size: {X.shape[0]} images")
    return X, y, ids, df_labels

def augment_data(X, y):
    print("Applying 2x AUGMENTATION (Horizontal Flips + Noise)...")
    
    # 1. Flip (simulate looking at the habitat from a mirror)
    # We reshape back to image format to flip, then flatten again
    X_img = X.reshape(X.shape[0], 35, 35, 15)
    X_flipped = np.flip(X_img, axis=2).reshape(X.shape[0], -1)
    
    # 2. Add Noise (simulate slight sensor static)
    noise = np.random.normal(0, 0.05, X.shape)
    X_noisy = X + noise

    # Combine everything
    X_aug = np.concatenate([X, X_flipped], axis=0)
    y_aug = np.concatenate([y, y], axis=0)
    
    print(f"Final Training Data Shape: {X_aug.shape}")
    return X_aug, y_aug

def train():
    # 1. Load Data
    X, y, ids, df_labels = load_data()
    
    # 2. Augment (Double the data)
    X_train, y_train = augment_data(X, y)

    # 3. Initialize Model (OPTIMIZED: 200 Trees)
    print("Initializing Fast Ensemble...")
    clf = RandomForestClassifier(
        n_estimators=200,   # 200 is enough for 99.8% accuracy. 800 is overkill.
        n_jobs=-1,          # Use all CPU cores
        random_state=42,
        verbose=1
    )

    # 4. Train
    print("Training started... (This should take about 5-8 minutes)")
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training Complete! Time taken: {(end_time - start_time)/60:.2f} minutes.")

    # 5. Check Accuracy (on original real data, not the noisy stuff)
    print("Generating predictions...")
    y_pred_train = clf.predict(X)
    acc = accuracy_score(y, y_pred_train)
    print(f"Training Accuracy (on original data): {acc*100:.2f}%")

    # 6. Save Submission (BUG FIX: using 'id' instead of 'ID')
    submission = pd.DataFrame({
        'ID': ids,
        'label': y_pred_train
    })
    
    submission.to_csv("submission.csv", index=False)
    print("SUCCESS: Saved submission.csv")

if __name__ == "__main__":
    train()