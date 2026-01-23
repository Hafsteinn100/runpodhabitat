import numpy as np
import pandas as pd

def create_submission():
    print("--- CREATING ENSEMBLE SUBMISSION ---")
    
    try:
        # 1. Load Probabilities
        probs_rf = np.load("probs_rf.npy")
        probs_eff = np.load("probs_efficient.npy")
        class_names = np.load("classes.npy", allow_pickle=True)
        
        # 2. Load Test IDs
        df_test = pd.read_csv("data/test/test.csv")
        # Handle 'id' vs 'ID' check
        ids = df_test['id'] if 'id' in df_test.columns else df_test['ID']

        print(f"Loaded RF Probs: {probs_rf.shape}")
        print(f"Loaded EfficientNet Probs: {probs_eff.shape}")

        # 3. Weighted Average
        # Give more weight to EfficientNet (Deep Learning usually generalizes better)
        # 0.6 for Deep Learning, 0.4 for Random Forest
        final_probs = (0.4 * probs_rf) + (0.6 * probs_eff)
        
        # 4. Get Final Labels
        final_indices = np.argmax(final_probs, axis=1)
        final_labels = class_names[final_indices]
        
        # 5. Save
        submission = pd.DataFrame({
            'ID': ids,
            'label': final_labels
        })
        
        submission.to_csv("submission_ensemble.csv", index=False)
        print("SUCCESS! Saved 'submission_ensemble.csv'. Upload this one!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you ran BOTH train_fast.py and train_efficient.py first!")

if __name__ == "__main__":
    create_submission()