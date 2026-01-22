import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier
from utils import load_data

def train():
    print("--- STARTING FINAL BOSS TRAINING (Augmented + Spatial) ---")
    
    # 1. Load Data (Now with 4x Augmentation)
    # This calls the load_data() function from utils.py above
    X, y = load_data()
    
    # 2. Define the Experts (Tuned for MAX performance)
    
    print("Initializing Random Forest (1200 trees)...")
    rf = RandomForestClassifier(
        n_estimators=1200,   # High tree count for the huge dataset
        max_depth=None,
        min_samples_leaf=1,  # Learn every detail
        n_jobs=-1,
        random_state=42
    )
    
    print("Initializing Extra Trees (1200 trees)...")
    et = ExtraTreesClassifier(
        n_estimators=1200,
        max_depth=None,
        min_samples_leaf=1,
        bootstrap=False,
        n_jobs=-1,
        random_state=42
    )
    
    print("Initializing Gradient Boosting (Deep Learning Mode)...")
    # This is the "Secret Weapon". It learns very slowly but very precisely.
    gb = HistGradientBoostingClassifier(
        max_iter=3000,       # 3000 rounds of correction
        learning_rate=0.01,  # Tiny steps for maximum accuracy
        max_depth=15,        # Deep logic
        l2_regularization=0.5,
        random_state=42
    )
    
    # 3. Voting
    print("Stacking models into VotingClassifier...")
    voting_model = VotingClassifier(
        estimators=[
            ('rf', rf), 
            ('et', et), 
            ('gb', gb)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # 4. Train
    print("Training on Augmented Dataset...")
    print("WARNING: This may take 15-20 minutes. Do not close it!")
    voting_model.fit(X, y)
    
    # 5. Save
    output_path = Path(__file__).parent / "model.joblib"
    print(f"Saving model to {output_path}...")
    joblib.dump(voting_model, output_path, compress=3)
    print("DONE! You are ready to win.")

if __name__ == "__main__":
    train()