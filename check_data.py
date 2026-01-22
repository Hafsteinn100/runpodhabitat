import numpy as np
from pathlib import Path

def check_stats():
    base_dir = Path(__file__).parent / "data"
    part1 = np.load(base_dir / "train" / "patches_part1.npy")
    # Just check part 1 to save time
    print(f"Shape: {part1.shape}")
    print(f"Min: {part1.min()}")
    print(f"Max: {part1.max()}")
    print(f"Mean: {part1.mean()}")
    print(f"Std: {part1.std()}")

if __name__ == "__main__":
    check_stats()
