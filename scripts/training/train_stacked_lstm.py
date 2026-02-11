"""
Train Stacked LSTM Models
Train models with 2, 3, 4, and 5 stacked LSTM layers.
Each configuration is trained independently.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.stacked_lstm import StackedLSTM, StackedLSTMConfig

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("outputs/checkpoints_stacked_lstm")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42
NUM_CLASSES = 9
PATIENCE = 10

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

def train_stacked_model(num_layers):
    """
    Train a stacked LSTM model with specified number of layers.
    
    Args:
        num_layers: Number of LSTM layers to stack (2, 3, 4, or 5)
    """
    
    print("\n" + "="*70)
    print(f"TRAINING {num_layers}-LAYER STACKED LSTM")
    print("="*70)
    
    # Get configuration for this depth
    config = StackedLSTMConfig.get_config(num_layers)
    
    # Set seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    # Load data
    print("\nLoading data...")
    X = np.load("outputs/sequences_X.npy")
    y = np.load("outputs/sequences_y.npy")
    video_ids = np.load("outputs/video_ids.npy", allow_pickle=True)
    
    print(f"Total sequences: {len(X)}")
    
    # Video-level split
    unique_videos = np.unique(video_ids)
    train_videos, val_videos = train_test_split(
        unique_videos, test_size=0.2, 
        random_state=RANDOM_SEED, shuffle=True
    )
    
    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    print(f"Training: {len(X_train)} sequences from {len(train_videos)} videos")
    print(f"Validation: {len(X_val)} sequences from {len(val_videos)} videos")
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
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
    print(f"\nInitializing {num_layers}-layer stacked LSTM...")
    model = StackedLSTM(
        input_size=X.shape[2],
        hidden_size=config['hidden_size'],
        num_classes=NUM_CLASSES,
        num_lstm_layers=num_layers,
        dropout=config['dropout']
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\nTraining for {config['num_epochs']} epochs...")
    best_val_acc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for deep networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                outputs = model(batch_X)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        # Calculate metrics
        val_acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Loss={avg_train_loss:.4f}, Acc={val_acc*100:.2f}%, "
              f"F1={f1:.4f}")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_f1 = f1
            patience_counter = 0
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'num_layers': num_layers,
                'config': config,
                'accuracy': val_acc,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'epoch': epoch + 1
            }
            
            checkpoint_path = CHECKPOINT_DIR / f"stacked_{num_layers}layer.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Best model saved! Acc={val_acc*100:.2f}%, F1={f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n✓ {num_layers}-layer model training complete!")
    print(f"  Best Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Best F1-Score: {best_f1:.4f}")
    
    return best_val_acc, best_f1

# ============================================================
# MAIN - TRAIN ALL CONFIGURATIONS
# ============================================================

def main():
    print("="*70)
    print("STACKED LSTM TRAINING - ALL CONFIGURATIONS")
    print("="*70)
    print("\nTraining models with 2, 3, 4, and 5 stacked LSTM layers.")
    print("Each layer learns hierarchical features.\n")
    
    results = {}
    
    # Train each configuration
    for num_layers in [2, 3, 4, 5]:
        acc, f1 = train_stacked_model(num_layers)
        results[f'{num_layers}-layer'] = {
            'accuracy': acc,
            'f1_score': f1
        }
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - ALL CONFIGURATIONS")
    print("="*70)
    
    print(f"\n{'Configuration':<20} {'Accuracy':<15} {'F1-Score'}")
    print("-" * 50)
    for config_name, metrics in results.items():
        print(f"{config_name:<20} {metrics['accuracy']*100:<14.2f}% {metrics['f1_score']:.4f}")
    
    # Save summary
    np.save(CHECKPOINT_DIR / "training_summary.npy", results)
    print(f"\n✓ Summary saved to: {CHECKPOINT_DIR / 'training_summary.npy'}")
    print(f"✓ All models saved to: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)