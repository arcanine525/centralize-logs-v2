"""
Prepare training data from WebAttack-CVSSMetrics HuggingFace dataset.

Downloads raw Apache logs, extracts features, and creates training CSV.

Dataset: https://huggingface.co/datasets/chYassine/WebAttack-CVSSMetrics

Usage:
    python -m ddos.prepare_data --output dataset/train.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import random
import sys

# Handle both module and direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ddos.features import parse_log, extract_features
    from ddos.config import TIME_WINDOW, FEATURE_NAMES
else:
    from .features import parse_log, extract_features
    from .config import TIME_WINDOW, FEATURE_NAMES


def download_dataset():
    """Download WebAttack-CVSSMetrics from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' library...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
        from datasets import load_dataset
    
    print("Downloading WebAttack-CVSSMetrics dataset...")
    ds = load_dataset("chYassine/WebAttack-CVSSMetrics")
    df = ds['data'].to_pandas()
    print(f"  Downloaded {len(df)} rows")
    return df


def create_training_samples(
    df: pd.DataFrame,
    window_size: int = 10,
    samples_per_class: int = 5000
) -> pd.DataFrame:
    """
    Create training samples by grouping logs into windows.
    
    Args:
        df: Raw dataset with '_raw' and 'Type' columns
        window_size: Number of logs per window
        samples_per_class: Max samples per class (normal/attack)
    
    Returns:
        DataFrame with features and labels
    """
    print(f"\nCreating training samples (window_size={window_size})...")
    
    # Separate normal and attack logs
    normal_logs = df[df['Type'].isna()]['_raw'].tolist()
    attack_logs = df[df['Type'].notna()]['_raw'].tolist()
    
    print(f"  Normal logs: {len(normal_logs)}")
    print(f"  Attack logs: {len(attack_logs)}")
    
    samples = []
    
    # Create normal samples (windows of normal traffic only)
    print("  Creating normal samples...")
    random.shuffle(normal_logs)
    for i in range(0, min(len(normal_logs), samples_per_class * window_size), window_size):
        window = normal_logs[i:i + window_size]
        if len(window) < window_size // 2:
            continue
        
        parsed = [parse_log(log) for log in window]
        parsed = [p for p in parsed if p is not None]
        
        if len(parsed) >= 3:
            features = extract_features(parsed, TIME_WINDOW)
            samples.append({
                **{name: features[i] for i, name in enumerate(FEATURE_NAMES)},
                'label': 0
            })
    
    n_normal = len(samples)
    print(f"    Created {n_normal} normal samples")
    
    # Create attack samples (windows with attack traffic)
    print("  Creating attack samples...")
    random.shuffle(attack_logs)
    
    # Mix attack logs with some normal logs to simulate realistic attack windows
    for i in range(0, min(len(attack_logs), samples_per_class * window_size // 2), window_size // 2):
        # Get attack logs for this window
        attack_window = attack_logs[i:i + window_size // 2]
        
        # Add some normal logs to mix
        normal_mix = random.sample(normal_logs, min(window_size // 2, len(normal_logs)))
        window = attack_window + normal_mix
        random.shuffle(window)
        
        parsed = [parse_log(log) for log in window]
        parsed = [p for p in parsed if p is not None]
        
        if len(parsed) >= 3:
            features = extract_features(parsed, TIME_WINDOW)
            samples.append({
                **{name: features[i] for i, name in enumerate(FEATURE_NAMES)},
                'label': 1
            })
    
    n_attack = len(samples) - n_normal
    print(f"    Created {n_attack} attack samples")
    
    # Create DataFrame
    result_df = pd.DataFrame(samples)
    
    # Balance classes
    n_min = min(n_normal, n_attack)
    if n_normal != n_attack:
        print(f"\n  Balancing classes to {n_min} each...")
        normal_samples = result_df[result_df['label'] == 0].sample(n=n_min, random_state=42)
        attack_samples = result_df[result_df['label'] == 1].sample(n=min(n_attack, n_min), random_state=42)
        result_df = pd.concat([normal_samples, attack_samples]).sample(frac=1, random_state=42)
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from HuggingFace dataset")
    parser.add_argument("--output", default="dataset/webattack_train.csv", help="Output CSV path")
    parser.add_argument("--window-size", type=int, default=10, help="Logs per window")
    parser.add_argument("--samples", type=int, default=5000, help="Max samples per class")
    args = parser.parse_args()
    
    print("=" * 60)
    print("WebAttack-CVSSMetrics Data Preparation")
    print("=" * 60)
    
    # Download dataset
    df = download_dataset()
    
    # Show attack type distribution
    print("\nAttack distribution:")
    attack_counts = df['Type'].value_counts(dropna=False)
    for attack_type, count in attack_counts.items():
        name = attack_type if pd.notna(attack_type) else "Normal"
        print(f"  {name}: {count}")
    
    # Create training samples
    train_df = create_training_samples(
        df,
        window_size=args.window_size,
        samples_per_class=args.samples
    )
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"Total samples: {len(train_df)}")
    print(f"  Normal: {sum(train_df['label'] == 0)}")
    print(f"  Attack: {sum(train_df['label'] == 1)}")
    print(f"Features: {len(FEATURE_NAMES)}")
    print(f"\nNext step:")
    print(f"  python -m ddos.train --data {output_path}")


if __name__ == "__main__":
    main()

