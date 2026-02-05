"""
Train Baseline LSTM Model
Independent training script for single baseline model using same hyperparameters as ensemble.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.baseline_lstm import BaselineLSTM

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 9
CHECKPOINT_DIR = Path("outputs/checkpoint_baseline")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# Hyperparameters matched to ensemble
RANDOM_SEED = 42
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
DROPOUT = 0.5
NUM_LAYERS = 2
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 15  # Early stopping

print(f"Device: {DEVICE}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")

# ============================================================
# DATASET CLASS
# ============================================================

class SequenceDataset(Dataset):
    """Dataset for hand landmark sequences"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_baseline_model():
    """Train a single baseline LSTM model with ensemble hyperparameters"""
    
    print("\n" + "="*70)
    print("BASELINE LSTM MODEL TRAINING")
    print("="*70)
    
    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    # Load preprocessed data
    X = np.load("outputs/sequences_X.npy")
    y = np.load("outputs/sequences_y.npy")
    video_ids = np.load("outputs/video_ids.npy", allow_pickle=True)
    
    # Video-level train-validation split
    unique_videos = np.unique(video_ids)
    train_videos, val_videos = train_test_split(
        unique_videos, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )
    
    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    # Create datasets and dataloaders
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    model = BaselineLSTM(
        input_size=X.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Loss function, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                preds = outputs.argmax(1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early stopping & checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "baseline_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {CHECKPOINT_DIR / 'baseline_model.pt'}")
    
    return best_val_acc

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    best_acc = train_baseline_model()
    print(f"\n✓ Training complete! Best validation accuracy: {best_acc:.2f}%")
