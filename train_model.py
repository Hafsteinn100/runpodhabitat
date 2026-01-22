import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier
from utils import load_data

def train():
    print("--- STARTING GOD MODE TRAINING (RunPod Edition) ---")
    
    # 1. Load Data (4x Augmentation)
    # This creates ~20,000 images with 330 features each.
    X, y = load_data()
    
    print(f"Training Data Shape: {X.shape}")
    print("Initializing Massive Ensemble...")

    # 2. Define the Experts (Unleashed for Cloud)
    
    # Expert 1: Random Forest
    # 2000 trees is massive, but RunPod RAM can handle it.
    rf = RandomForestClassifier(
        n_estimators=2000,   
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,          # Use ALL RunPod vCPUs
        random_state=42
    )
    
    # Expert 2: Extra Trees
    # 2000 trees ensures we capture every single texture variation.
    et = ExtraTreesClassifier(
        n_estimators=2000,
        max_depth=None,
        min_samples_leaf=1,
        bootstrap=False,
        n_jobs=-1,          # Use ALL RunPod vCPUs
        random_state=42
    )
    
    # Expert 3: Gradient Boosting
    # 5000 iterations @ 0.01 learning rate = Extreme Precision.
    gb = HistGradientBoostingClassifier(
        max_iter=5000,       # Learn forever
        learning_rate=0.01,  # Learn carefully
        max_depth=15,
        l2_regularization=0.5,
        random_state=42
    )
    
    # 3. Voting
    print("Stacking models into Parallel VotingClassifier...")
    
    # CLOUD SETTING: n_jobs=-1
    # We train all 3 monster models AT THE SAME TIME.
    voting_model = VotingClassifier(
        estimators=[
            ('rf', rf), 
            ('et', et), 
            ('gb', gb)
        ],
        voting='soft',
        n_jobs=-1  # Parallel training enabled!
    )
    
    # 4. Train
    print("Training started... (This utilizes 100% of the Cloud CPU)")
    voting_model.fit(X, y)
    
    # 5. Save
    output_path = Path(__file__).parent / "model.joblib"
    print(f"Saving model to {output_path}...")
    joblib.dump(voting_model, output_path, compress=3)
    print("DONE! Download 'model.joblib' and win this thing.")

if __name__ == "__main__":
    train()