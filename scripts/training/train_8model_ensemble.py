"""
Train 8-Model Specialized Ensemble
Each model is an expert for one WHO step vs background.
Models are trained independently (no for-loop) for clarity.
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

from src.models.specialized_lstm import SpecializedLSTM

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("outputs/checkpoints_8model_ensemble")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# Training hyperparameters
RANDOM_SEED = 42
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
DROPOUT = 0.4
BATCH_SIZE = 32
NUM_EPOCHS = 50
PATIENCE = 10

print(f"Device: {DEVICE}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")

# ============================================================
# DATASET CLASS
# ============================================================

class BinarySequenceDataset(Dataset):
    """Dataset for binary classification (target step vs background)"""
    def __init__(self, X, y, target_step):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Convert to binary: 1 if target_step, 0 otherwise
        self.y = torch.tensor((y == target_step).astype(int), dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_specialized_model(model_id, target_step, step_name):
    """
    Train a single specialized model.
    
    Args:
        model_id: Model identifier (1-8)
        target_step: WHO step this model recognizes (1-8)
        step_name: Description of the step
    """
    
    print("\n" + "="*70)
    print(f"TRAINING MODEL {model_id}: {step_name}")
    print("="*70)
    
    # Set seeds
    torch.manual_seed(RANDOM_SEED + model_id)  # Different seed per model
    np.random.seed(RANDOM_SEED + model_id)
    
    # Load data
    print("\nLoading data...")
    X = np.load("outputs/sequences_X.npy")
    y = np.load("outputs/sequences_y.npy")
    video_ids = np.load("outputs/video_ids.npy", allow_pickle=True)
    
    # Video-level split
    unique_videos = np.unique(video_ids)
    train_videos, val_videos = train_test_split(
        unique_videos, test_size=0.2, 
        random_state=RANDOM_SEED + model_id, shuffle=True
    )
    
    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    # Create binary datasets
    train_dataset = BinarySequenceDataset(X_train, y_train, target_step)
    val_dataset = BinarySequenceDataset(X_val, y_val, target_step)
    
    # Count class distribution
    train_positive = (y_train == target_step).sum()
    train_negative = len(y_train) - train_positive
    val_positive = (y_val == target_step).sum()
    val_negative = len(y_val) - val_positive
    
    print(f"\nTarget Step: Step {target_step}")
    print(f"Training: {train_positive} positive, {train_negative} negative")
    print(f"Validation: {val_positive} positive, {val_negative} negative")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, 
                           shuffle=False, num_workers=0)
    
    # Initialize model
    model = SpecializedLSTM(
        input_size=X.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Calculate class weights for imbalanced data
    total = train_positive + train_negative
    weight_positive = total / (2.0 * train_positive) if train_positive > 0 else 1.0
    weight_negative = total / (2.0 * train_negative) if train_negative > 0 else 1.0
    class_weights = torch.tensor([weight_negative, weight_positive], dtype=torch.float32).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\nTraining Model {model_id}...")
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
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
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Loss={avg_train_loss:.4f}, "
              f"Acc={accuracy*100:.2f}%, Prec={precision:.4f}, "
              f"Rec={recall:.4f}, F1={f1:.4f}")
        
        scheduler.step(f1)
        
        # Save best model based on F1-score
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            checkpoint_path = CHECKPOINT_DIR / f"model_{model_id}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_step': target_step,
                'step_name': step_name,
                'best_f1': best_f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }, checkpoint_path)
            print(f"  ✓ Best F1: {f1:.4f} - Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n✓ Model {model_id} training complete!")
    print(f"  Best F1-Score: {best_f1:.4f}")
    
    return best_f1, accuracy, precision, recall

# ============================================================
# TRAIN ALL 8 MODELS
# ============================================================

def main():
    print("="*70)
    print("8-MODEL SPECIALIZED ENSEMBLE TRAINING")
    print("="*70)
    print("\nEach model is an expert for one WHO step vs background.")
    print("Models are trained independently for maximum clarity.\n")
    
    results = {}
    
    # ============================================================
    # MODEL 1: Step 1 (Palm to Palm)
    # ============================================================
    f1, acc, prec, rec = train_specialized_model(
        model_id=1,
        target_step=1,
        step_name="Step 1: Palm to Palm"
    )
    results['Model 1'] = {'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}
    
    # ============================================================
    # MODEL 2: Step 2 (Right Palm over Left Dorsum)
    # ============================================================
    f1, acc, prec, rec = train_specialized_model(
        model_id=2,
        target_step=2,
        step_name="Step 2: Right Palm over Left Dorsum"
    )
    results['Model 2'] = {'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}
    
    # ============================================================
    # MODEL 3: Step 3 (Left Palm over Right Dorsum)
    # ============================================================
    f1, acc, prec, rec = train_specialized_model(
        model_id=3,
        target_step=3,
        step_name="Step 3: Left Palm over Right Dorsum"
    )
    results['Model 3'] = {'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}
    
    # ============================================================
    # MODEL 4: Step 4 (Fingers Interlaced)
    # ============================================================
    f1, acc, prec, rec = train_specialized_model(
        model_id=4,
        target_step=4,
        step_name="Step 4: Fingers Interlaced"
    )
    results['Model 4'] = {'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}
    
    # ============================================================
    # MODEL 5: Step 5 (Backs of Fingers)
    # ============================================================
    f1, acc, prec, rec = train_specialized_model(
        model_id=5,
        target_step=5,
        step_name="Step 5: Backs of Fingers"
    )
    results['Model 5'] = {'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}
    
    # ============================================================
    # MODEL 6: Step 6 (Rotational Rubbing of Thumbs)
    # ============================================================
    f1, acc, prec, rec = train_specialized_model(
        model_id=6,
        target_step=6,
        step_name="Step 6: Rotational Rubbing of Thumbs"
    )
    results['Model 6'] = {'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}
    
    # ============================================================
    # MODEL 7: Step 7 (Rotational Rubbing of Fingertips)
    # ============================================================
    f1, acc, prec, rec = train_specialized_model(
        model_id=7,
        target_step=7,
        step_name="Step 7: Rotational Rubbing of Fingertips"
    )
    results['Model 7'] = {'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}
    
    # ============================================================
    # MODEL 8: Step 8 (Rotational Rubbing of Wrists)
    # ============================================================
    f1, acc, prec, rec = train_specialized_model(
        model_id=8,
        target_step=8,
        step_name="Step 8: Rotational Rubbing of Wrists"
    )
    results['Model 8'] = {'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE - ALL 8 MODELS")
    print("="*70)
    
    print(f"\n{'Model':<15} {'F1-Score':<12} {'Accuracy':<12} {'Precision':<12} {'Recall'}")
    print("-" * 65)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['f1']:<12.4f} {metrics['accuracy']*100:<11.2f}% "
              f"{metrics['precision']:<12.4f} {metrics['recall']:.4f}")
    
    # Save summary
    np.save(CHECKPOINT_DIR / "training_summary.npy", results)
    print(f"\n✓ Training summary saved to: {CHECKPOINT_DIR / 'training_summary.npy'}")
    print(f"✓ All models saved to: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)