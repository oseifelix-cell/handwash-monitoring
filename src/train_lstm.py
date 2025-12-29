"""
Train an ensemble of 5 models with different:
- Random seeds
- Train/val splits
- Slightly different hyperparameters

Then average their predictions for final accuracy.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.lstm_model import HandwashLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("outputs/checkpoints_ensemble")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
X = np.load("outputs/sequences_X.npy")
y = np.load("outputs/sequences_y.npy")
video_ids = np.load("outputs/video_ids.npy")

NUM_CLASSES = len(np.unique(y))


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_single_model(model_id, seed, hidden_size, lr, dropout):
    """Train one model in the ensemble"""
    print(f"\n{'='*60}")
    print(f"🤖 Training Model {model_id}/5")
    print(f"   Seed: {seed} | Hidden: {hidden_size} | LR: {lr} | Dropout: {dropout}")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Split with this seed
    unique_videos = np.unique(video_ids)
    train_videos, val_videos = train_test_split(
        unique_videos, test_size=0.2, random_state=seed, shuffle=True
    )
    
    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)
    
    train_ds = SequenceDataset(X[train_mask], y[train_mask])
    val_ds = SequenceDataset(X[val_mask], y[val_mask])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=0)
    
    # Class weights
    weights = compute_class_weight("balanced", classes=np.unique(y), y=y[train_mask])
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    # Model
    model = HandwashLSTM(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        num_classes=NUM_CLASSES,
        num_layers=2,
        dropout=dropout
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    
    # Train
    best_val = 0
    patience = 0
    
    for epoch in range(100):
        # Training
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                preds = out.argmax(1)
                v_correct += (preds == yb).sum().item()
                v_total += yb.size(0)
        
        val_acc = 100 * v_correct / v_total
        scheduler.step(val_acc)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val:
            best_val = val_acc
            patience = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"model_{model_id}.pt")
        else:
            patience += 1
            if patience >= 15:
                break
    
    print(f"  ✅ Model {model_id} Best Val Acc: {best_val:.2f}%")
    return best_val, val_videos


# Train 5 models with different configurations
configs = [
    (42, 128, 1e-3, 0.3),    # Model 1: Baseline
    (123, 112, 8e-4, 0.4),   # Model 2: Smaller, more dropout
    (777, 144, 1.2e-3, 0.35), # Model 3: Larger, moderate dropout
    (2024, 128, 7e-4, 0.45),  # Model 4: Conservative
    (999, 96, 9e-4, 0.5),     # Model 5: Small, high dropout
]

print("\n" + "="*60)
print("🎯 ENSEMBLE TRAINING - 5 Models")
print("="*60)

model_accs = []
val_video_sets = []

for i, (seed, hidden, lr, dropout) in enumerate(configs, 1):
    acc, val_vids = train_single_model(i, seed, hidden, lr, dropout)
    model_accs.append(acc)
    val_video_sets.append(val_vids)

# Test ensemble on a common validation set
print("\n" + "="*60)
print("🔮 ENSEMBLE PREDICTION")
print("="*60)

# Use validation set from first model
unique_videos = np.unique(video_ids)
_, val_videos = train_test_split(unique_videos, test_size=0.2, random_state=42, shuffle=True)
val_mask = np.isin(video_ids, val_videos)

X_val = torch.tensor(X[val_mask], dtype=torch.float32).to(DEVICE)
y_val = y[val_mask]

# Load all models and get predictions
all_predictions = []

for i in range(1, 6):
    model = HandwashLSTM(
        input_size=X.shape[2],
        hidden_size=configs[i-1][1],
        num_classes=NUM_CLASSES,
        num_layers=2,
        dropout=configs[i-1][3]
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"model_{i}.pt"))
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_val)
        probs = torch.softmax(outputs, dim=1)
        all_predictions.append(probs.cpu().numpy())

# Average predictions
avg_predictions = np.mean(all_predictions, axis=0)
ensemble_preds = avg_predictions.argmax(axis=1)

# Calculate ensemble accuracy
ensemble_acc = 100 * (ensemble_preds == y_val).sum() / len(y_val)

print(f"\n📊 Individual Model Accuracies:")
for i, acc in enumerate(model_accs, 1):
    print(f"  Model {i}: {acc:.2f}%")

print(f"\n🏆 ENSEMBLE ACCURACY: {ensemble_acc:.2f}%")
print(f"   Improvement: +{ensemble_acc - max(model_accs):.2f}% over best single model")
print("="*60)