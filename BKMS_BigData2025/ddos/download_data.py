"""Quick download and save script."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60, flush=True)
print("Downloading WebAttack-CVSSMetrics from HuggingFace", flush=True)
print("=" * 60, flush=True)

from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Download
print("\n[1/4] Downloading dataset...", flush=True)
ds = load_dataset("chYassine/WebAttack-CVSSMetrics")
df = ds['data'].to_pandas()
print(f"      Downloaded {len(df)} rows", flush=True)

# Show distribution
print("\n[2/4] Attack distribution:", flush=True)
for attack_type, count in df['Type'].value_counts(dropna=False).items():
    name = attack_type if pd.notna(attack_type) else "Normal"
    print(f"      {name}: {count}", flush=True)

# Save raw data
print("\n[3/4] Saving raw data...", flush=True)
output_dir = Path("dataset")
output_dir.mkdir(exist_ok=True)
raw_path = output_dir / "webattack_raw.csv"
df.to_csv(raw_path, index=False)
print(f"      Saved to: {raw_path}", flush=True)

# Create simple labeled dataset (for quick training)
print("\n[4/4] Creating labeled dataset...", flush=True)
df['label'] = df['Type'].apply(lambda x: 0 if pd.isna(x) else 1)
labeled = df[['_raw', 'Type', 'label']].copy()
labeled_path = output_dir / "webattack_labeled.csv"
labeled.to_csv(labeled_path, index=False)
print(f"      Saved to: {labeled_path}", flush=True)
print(f"      Normal: {sum(df['label']==0)}, Attack: {sum(df['label']==1)}", flush=True)

print("\n" + "=" * 60, flush=True)
print("DOWNLOAD COMPLETE!", flush=True)
print("=" * 60, flush=True)
print(f"\nNext: Run feature extraction and training", flush=True)

