"""Train model from raw WebAttack dataset."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import random
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ddos.features import parse_log, extract_features
from ddos.config import TIME_WINDOW, FEATURE_NAMES, INPUT_DIM, HIDDEN_LAYERS, DROPOUT_RATE
from ddos.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE
from ddos.config import MODEL_PATH, TORCHSCRIPT_PATH, MODEL_DIR
from ddos.model import ApacheDDoSModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 60, flush=True)
print("Web Attack Detection - Training", flush=True)
print("=" * 60, flush=True)

# Load raw data
raw_path = Path("dataset/webattack_labeled.csv")
print(f"\n[1/6] Loading {raw_path}...", flush=True)
df = pd.read_csv(raw_path)
print(f"      Total logs: {len(df)}", flush=True)

# Separate normal and attack logs
normal_logs = df[df['label'] == 0]['_raw'].tolist()
attack_logs = df[df['label'] == 1]['_raw'].tolist()
print(f"      Normal: {len(normal_logs)}, Attack: {len(attack_logs)}", flush=True)

# Create training samples by grouping logs
print("\n[2/6] Creating training samples (window_size=10)...", flush=True)
window_size = 10
samples = []

random.seed(42)
random.shuffle(normal_logs)
random.shuffle(attack_logs)

# Normal samples - USE ALL DATA
print("      Creating normal samples...", flush=True)
for i in range(0, len(normal_logs), window_size):
    window = normal_logs[i:i + window_size]
    if len(window) < 5:
        continue
    parsed = [parse_log(log) for log in window]
    parsed = [p for p in parsed if p is not None]
    if len(parsed) >= 3:
        features = extract_features(parsed, TIME_WINDOW)
        samples.append((*features, 0))

n_normal = len(samples)
print(f"      Created {n_normal} normal samples", flush=True)

# Attack samples - USE ALL DATA (mix attack with some normal)
print("      Creating attack samples...", flush=True)
for i in range(0, len(attack_logs), window_size // 2):
    attack_window = attack_logs[i:i + window_size // 2]
    normal_mix = random.sample(normal_logs, min(window_size // 2, len(normal_logs)))
    window = attack_window + normal_mix
    random.shuffle(window)
    
    parsed = [parse_log(log) for log in window]
    parsed = [p for p in parsed if p is not None]
    if len(parsed) >= 3:
        features = extract_features(parsed, TIME_WINDOW)
        samples.append((*features, 1))

n_attack = len(samples) - n_normal
print(f"      Created {n_attack} attack samples", flush=True)

# Create DataFrame
columns = FEATURE_NAMES + ['label']
train_df = pd.DataFrame(samples, columns=columns)

# Balance classes
n_min = min(n_normal, n_attack)
print(f"\n[3/6] Balancing to {n_min} samples per class...", flush=True)
normal_samples = train_df[train_df['label'] == 0].sample(n=n_min, random_state=42)
attack_samples = train_df[train_df['label'] == 1].sample(n=min(n_attack, n_min), random_state=42)
train_df = pd.concat([normal_samples, attack_samples]).sample(frac=1, random_state=42)

# Prepare data
X = train_df[FEATURE_NAMES].values
y = train_df['label'].values
print(f"      Total samples: {len(X)}", flush=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
print(f"      Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}", flush=True)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val_scaled)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

# Model
print("\n[4/6] Training model...", flush=True)
model = ApacheDDoSModel(INPUT_DIM, HIDDEN_LAYERS, DROPOUT_RATE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_loss = float('inf')
patience_counter = 0
best_state = None

print(f"      Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}", flush=True)
print("-" * 60, flush=True)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
        val_acc = ((val_outputs >= 0.5).float() == y_val_t).float().mean().item()
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}", flush=True)
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}", flush=True)
        break

model.load_state_dict(best_state)

# Evaluate
print("\n[5/6] Evaluating...", flush=True)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test)

model.eval()
with torch.no_grad():
    outputs = model(X_test_t).squeeze()
    preds = (outputs >= 0.5).float()

tp = ((preds == 1) & (y_test_t == 1)).sum().item()
tn = ((preds == 0) & (y_test_t == 0)).sum().item()
fp = ((preds == 1) & (y_test_t == 0)).sum().item()
fn = ((preds == 0) & (y_test_t == 1)).sum().item()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)

print(f"      Accuracy:  {accuracy:.4f}", flush=True)
print(f"      Precision: {precision:.4f}", flush=True)
print(f"      Recall:    {recall:.4f}", flush=True)
print(f"      F1-Score:  {f1:.4f}", flush=True)
print(f"      Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)

# Save
print("\n[6/6] Saving model...", flush=True)
MODEL_DIR.mkdir(exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': INPUT_DIM,
    'hidden_layers': HIDDEN_LAYERS,
    'dropout_rate': DROPOUT_RATE,
}, MODEL_PATH)
print(f"      PyTorch: {MODEL_PATH}", flush=True)

model.eval()
traced = torch.jit.trace(model, torch.randn(1, INPUT_DIM))
traced.save(str(TORCHSCRIPT_PATH))
print(f"      TorchScript: {TORCHSCRIPT_PATH}", flush=True)

scaler_path = MODEL_DIR / "scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"      Scaler: {scaler_path}", flush=True)

print("\n" + "=" * 60, flush=True)
print("TRAINING COMPLETE!", flush=True)
print("=" * 60, flush=True)

