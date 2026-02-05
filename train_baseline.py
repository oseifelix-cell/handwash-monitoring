"""
Train Baseline LSTM Model
Independent training script for single baseline model using optimal hyperparameters.
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

# Optimal hyperparameters (similar to best ensemble configuration)
RANDOM_SEED = 42
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
DROPOUT = 0.4
NUM_LAYERS = 2
BATCH_SIZE = 32
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience

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
    """Train a single baseline LSTM model with optimal hyperparameters"""
    
    print("\n" + "="*70)
    print("BASELINE LSTM MODEL TRAINING")
    print("="*70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Configuration: hidden={HIDDEN_SIZE}, lr={LEARNING_RATE}, dropout={DROPOUT}")
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X = np.load("outputs/sequences_X.npy")
    y = np.load("outputs/sequences_y.npy")
    video_ids = np.load("outputs/video_ids.npy", allow_pickle=True)
    
    print(f"Total sequences: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Video-level train-validation split (80-20)
    unique_videos = np.unique(video_ids)
    print(f"\nTotal videos: {len(unique_videos)}")
    
    train_videos, val_videos = train_test_split(
        unique_videos, 
        test_size=0.2, 
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"Training videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    
    # Create masks for train/val split
    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)
    
    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    print(f"\nTraining sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    
    # Class distribution
    print("\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Create datasets and dataloaders
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = BaselineLSTM(
        input_size=X.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING STARTED")
    print(f"{'='*70}\n")
    
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch_X, batch_y in train_pbar:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for batch_X, batch_y in val_pbar:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_path = CHECKPOINT_DIR / "baseline_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ New best model saved! (Accuracy: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{PATIENCE})")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {CHECKPOINT_DIR / 'baseline_model.pt'}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'hyperparameters': {
            'hidden_size': HIDDEN_SIZE,
            'learning_rate': LEARNING_RATE,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'random_seed': RANDOM_SEED
        }
    }
    
    np.save(CHECKPOINT_DIR / "training_history.npy", history)
    print(f"Training history saved to: {CHECKPOINT_DIR / 'training_history.npy'}")
    
    return best_val_acc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        best_acc = train_baseline_model()
        print(f"\n✓ Training successful! Best accuracy: {best_acc:.2f}%")
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)