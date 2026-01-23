
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
import os

# --- CONFIGURATION ---
SUBMISSION_FILE = "submission.csv"

def load_data():
    print("Loading data...")
    
    # Robust path finding (Fixed for current env)
    possible_roots = ["data", "data/train", "."] 
    p1_path = None
    p2_path = None
    csv_path = None
    
    for root in possible_roots:
        p1 = os.path.join(root, "patches_part1.npy")
        if os.path.exists(p1):
            p1_path = p1
            p2_path = os.path.join(root, "patches_part2.npy")
            print(f"Found data in: {root}")
            break
            
    if p1_path is None:
        raise FileNotFoundError("Could not find patches_part1.npy")

    # Finalize paths
    X_part1 = np.load(p1_path)
    X_part2 = np.load(p2_path)
    X = np.concatenate([X_part1, X_part2], axis=0)

    # Flatten images (from 4D to 2D)
    X = X.reshape(X.shape[0], -1)

    # Find CSV
    for root in possible_roots:
        c = os.path.join(root, "train.csv")
        if os.path.exists(c):
            csv_path = c
            break
            
    if csv_path is None:
         # Try specific
         if os.path.exists("data/train.csv"): csv_path = "data/train.csv"
         else: raise FileNotFoundError("Could not find train.csv")

    df_labels = pd.read_csv(csv_path)
    # Check for label column (Fixed to vistgerd_idx)
    if 'vistgerd_idx' in df_labels.columns:
         y = df_labels['vistgerd_idx'].values
    elif 'label' in df_labels.columns:
        y = df_labels['label'].values
    else:
        # Fallback search
        cols = df_labels.columns
        print(f"Columns found: {cols}")
        if 'vistgerd_idx' in cols: y = df_labels['vistgerd_idx'].values
        else: raise KeyError("Could not find 'vistgerd_idx' or 'label' column.")

    print(f"Original dataset size: {X.shape[0]} images")
    return X, y, df_labels

def augment_data(X, y):
    print("Applying 2x AUGMENTATION (Horizontal Flips)...")
    
    # Reshape to (N, 15, 35, 35) assuming channels first or last?
    # Based on previous robust check, we assume (N, 15, 35, 35) if channels=15 is dim 1
    # or (N, 35, 35, 15) if dim 3.
    # Sentinel data is usually (15, 35, 35).
    # X is flattened here. We need to unflatten.
    # 15*35*35 = 18375
    
    X_img = X.reshape(X.shape[0], 15, 35, 35)
    
    # Flip axis 3 (width)
    X_flipped = np.flip(X_img, axis=3).reshape(X.shape[0], -1) 
    
    X_aug = np.concatenate([X, X_flipped], axis=0)
    y_aug = np.concatenate([y, y], axis=0)
    
    print(f"Final Training Data Shape: {X_aug.shape}")
    return X_aug, y_aug

def train():
    # 1. Load Data
    X, y, df_labels = load_data()
    
    # Encode labels just in case
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # 2. Augment
    X_train, y_train_enc = augment_data(X, y_enc)

    # 3. Define Models (Restored Short Config: 200 trees)
    print("Initializing Ensemble (RF + ET + HGB)...")
    
    # Random Forest: Reliable, robust
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
    
    # 6. Predict on original data (sanity check)
    print("Generating predictions on training set (sanity check)...")
    preds_enc = model.predict(X) 
    preds = le.inverse_transform(preds_enc)
    
    acc = accuracy_score(y, preds)
    print(f"Training Accuracy (on original data): {acc * 100:.2f}%")
    
    # 7. GENERATE SUBMISSION FOR TEST DATA (If available)
    print("Generating submission.csv for Test Data...")
    
    # Try to find test data
    test_roots = ["data", "data/test", "."]
    test_path = None
    for root in test_roots:
        t = os.path.join(root, "patches_test.npy")
        if os.path.exists(t):
            test_path = t
            print(f"Found test data: {test_path}")
            break
            
    if test_path:
        X_test_raw = np.load(test_path)
        X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)
        
        # Predict
        test_preds_enc = model.predict(X_test)
        test_preds = le.inverse_transform(test_preds_enc)
        
        # Load test csv for IDs
        # Try finding test.csv
        test_csv_path = None
        for root in test_roots:
             c = os.path.join(root, "test.csv")
             if os.path.exists(c):
                 test_csv_path = c
                 break
        
        ids = None
        if test_csv_path:
            df_test = pd.read_csv(test_csv_path)
            # Check ID column
            if 'ID' in df_test.columns: ids = df_test['ID']
            elif 'id' in df_test.columns: ids = df_test['id']
            else: ids = range(len(test_preds))
        else:
            ids = range(len(test_preds))
            
        submission = pd.DataFrame({
            'ID': ids,
            'label': test_preds
        })
        submission.to_csv(SUBMISSION_FILE, index=False)
        print(f"SUCCESS: Submission saved to {SUBMISSION_FILE}")
        
    else:
        print("WARNING: 'patches_test.npy' not found! Cannot generate submission file.")
        print("Please upload 'patches_test.npy' to the 'data/' folder.")

if __name__ == "__main__":
    train()