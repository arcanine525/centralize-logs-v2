"""
Training script for Apache Log DDoS Detection Model.

Expected CSV format:
- 16 feature columns (see config.FEATURE_NAMES)
- 1 label column: 'label' (0=normal, 1=ddos)

Usage:
    python -m ddos.train --data dataset/train.csv --epochs 100
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

from .config import (
    INPUT_DIM, HIDDEN_LAYERS, DROPOUT_RATE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE,
    MODEL_PATH, TORCHSCRIPT_PATH, MODEL_DIR, FEATURE_NAMES
)
from .model import ApacheDDoSModel


def load_dataset(csv_path: str):
    """Load and prepare dataset from CSV."""
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check if we have feature columns or need to use all except 'label'
    if 'label' in df.columns:
        y = df['label'].values
        X = df.drop(columns=['label']).values
    else:
        # Assume last column is label
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Normal: {sum(y==0)}, DDoS: {sum(y==1)}")
    
    return X, y


def train_model(
    X_train, y_train, X_val, y_val,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE
):
    """Train the MLP model."""
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    input_dim = X_train.shape[1]
    model = ApacheDDoSModel(input_dim, HIDDEN_LAYERS, DROPOUT_RATE)
    
    # Class weights for imbalanced data
    pos_weight = torch.tensor([sum(y_train == 0) / max(sum(y_train == 1), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use BCELoss since model outputs sigmoid
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Train
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
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_preds = (val_outputs >= 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    
    return model, scaler


def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model on test set."""
    X_test_scaled = scaler.transform(X_test)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t).squeeze()
        preds = (outputs >= 0.5).float()
    
    # Metrics
    tp = ((preds == 1) & (y_test_t == 1)).sum().item()
    tn = ((preds == 0) & (y_test_t == 0)).sum().item()
    fp = ((preds == 1) & (y_test_t == 0)).sum().item()
    fn = ((preds == 0) & (y_test_t == 1)).sum().item()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={tp}, FP={fp}")
    print(f"  FN={fn}, TN={tn}")
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def save_model(model, scaler, input_dim: int):
    """Save model in both PyTorch and TorchScript formats."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_layers': HIDDEN_LAYERS,
        'dropout_rate': DROPOUT_RATE,
    }, MODEL_PATH)
    print(f"\nPyTorch model saved: {MODEL_PATH}")
    
    # Save TorchScript model (for production)
    model.eval()
    example_input = torch.randn(1, input_dim)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(str(TORCHSCRIPT_PATH))
    print(f"TorchScript model saved: {TORCHSCRIPT_PATH}")
    
    # Save scaler
    scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Apache Log DDoS Detection Model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Apache Log DDoS Detection - Training")
    print("=" * 60)
    
    # Load data
    X, y = load_dataset(args.data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    
    # Train
    model, scaler = train_model(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Evaluate
    evaluate_model(model, X_test, y_test, scaler)
    
    # Save
    save_model(model, scaler, X_train.shape[1])
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

