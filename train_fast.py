import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time

# --- CONFIGURATION ---
DATA_DIR = "data"
PART1_FILE = os.path.join(DATA_DIR, "patches_part1.npy")
PART2_FILE = os.path.join(DATA_DIR, "patches_part2.npy")
LABELS_FILE = os.path.join(DATA_DIR, "train.csv")
SUBMISSION_FILE = "submission.csv"

def load_data():
    print("Loading raw data...")
    if not os.path.exists(PART1_FILE) or not os.path.exists(PART2_FILE):
        raise FileNotFoundError("Data parts missing! Check data/ folder.")
    
    # Load and combine parts
    p1 = np.load(PART1_FILE)
    p2 = np.load(PART2_FILE)
    X = np.concatenate([p1, p2], axis=0)
    
    # Load labels
    df = pd.read_csv(LABELS_FILE)
    y = df['Label'].values
    
    # Check alignment
    if len(X) != len(y):
        print(f"Warning: X has {len(X)} samples, y has {len(y)} samples. Truncating to match.")
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
    return X, y, df

def augment_data(X, y):
    """
    Applies 'Smart' Augmentation:
    1. Original Image (Flattened)
    2. Horizontal Flip (Left-Right) (Flattened)
    
    This doubles the dataset (2x).
    """
    print(f"Original dataset size: {len(X)} images")
    print("Applying 2x AUGMENTATION (Horizontal Flips + Noise)...")
    
    X_aug = []
    y_aug = []
    
    # Calculate noise scale (approx 2% of standard deviation)
    # Based on normalization.json: std is ~1208
    NOISE_SCALE = 25.0 
    
    for i in range(len(X)):
        # X[i] is (15, 35, 35)
        img_3d = X[i]
        label = y[i]
        
        # 1. Original (Flattened)
        X_aug.append(img_3d.ravel())
        y_aug.append(label)
        
        # 2. Horizontal Flip (preserving spatial structure before flatten)
        # Shape is (Channels, H, W) -> Flip on last axis (W)
        img_flipped = np.flip(img_3d, axis=2)
        
        # Add slight noise to the flipped version to improve robustness
        noise = np.random.normal(0, NOISE_SCALE, img_flipped.shape)
        img_flipped_noisy = img_flipped + noise
        
        X_aug.append(img_flipped_noisy.ravel())
        y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)

def train():
    print("--- STARTING SPEEDSTER TRAINING ---")
    
    # 1. Load
    X, y, df_labels = load_data() # X is (N, 15, 35, 35)
    
    # 2. Augment & Flatten
    # Returns (N*2, Features)
    X_train, y_train = augment_data(X, y)
    print(f"Final Training Data Shape: {X_train.shape}")
    
    # 3. Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    # 4. Define 'The Speedster' Ensemble
    print("Initializing Fast Ensemble...")
    
    # Random Forest: Reliable, Parallel (n_jobs=-1 uses all cores)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    
    # Extra Trees: Faster than Random Forest, reduces variance
    et = ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    
    # HistGradientBoosting: The Speed Demon (Sklearn's version of LightGBM)
    hgb = HistGradientBoostingClassifier(max_iter=150, random_state=42)
    
    # Voting Classifier: Combines them all
    model = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('et', et),
            ('hgb', hgb)
        ],
        voting='soft',
        n_jobs=-1  # Parallelize the voting process too
    )
    
    # 5. Train
    print("Training started... (This should be much faster)")
    start_time = time.time()
    model.fit(X_train, y_train_enc)
    print(f"Training Complete! Time taken: {round((time.time() - start_time)/60, 2)} minutes.")
    
    # 6. Predict on original data (for submission consistency check)
    # In a real scenario, you'd predict on a test set. 
    # Here we just generate the submission file format.
    print("Generating predictions...")
    
    # Note: We predict on the ORIGINAL X (no noise), but we need to ensure order matches submission
    # The competition usually requires predicting on a separate 'test.csv' or folder.
    # Since we don't have a test set file here, we assume we are just saving the model
    # or outputting training metrics. 
    
    # Create submission file (just mapping ids to labels)
    preds_enc = model.predict(X.reshape(len(X), -1))
    preds = le.inverse_transform(preds_enc)
    
    acc = accuracy_score(y, preds)
    print(f"Training Accuracy (on original data): {acc * 100:.2f}%")
    
    # Create submission file (just mapping ids to labels)
    submission = pd.DataFrame({
        'ID': df_labels['ID'],
        'Label': preds
    })
    
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")

if __name__ == "__main__":
    train()