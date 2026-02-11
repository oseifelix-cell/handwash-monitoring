"""
Evaluate Stacked LSTM Models
Comprehensive evaluation with accuracy, F1-score, precision, recall, and confusion matrix.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.stacked_lstm import StackedLSTM, StackedLSTMConfig

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 9
CHECKPOINT_DIR = Path("outputs/checkpoints_stacked_lstm")
RANDOM_SEED = 42

WHO_STEPS = {
    0: "Background/Not Washing",
    1: "Step 1: Palm to Palm",
    2: "Step 2: Right Palm over Left Dorsum",
    3: "Step 3: Left Palm over Right Dorsum",
    4: "Step 4: Fingers Interlaced",
    5: "Step 5: Backs of Fingers",
    6: "Step 6: Rotational Rubbing of Thumbs",
    7: "Step 7: Rotational Rubbing of Fingertips",
    8: "Step 8: Rotational Rubbing of Wrists"
}

print(f"Device: {DEVICE}")

# ============================================================
# EVALUATE SINGLE CONFIGURATION
# ============================================================

def evaluate_stacked_model(num_layers):
    """Evaluate a single stacked LSTM configuration"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {num_layers}-LAYER STACKED LSTM")
    print(f"{'='*70}")
    
    # Load checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"stacked_{num_layers}layer.pt"
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint['config']
    
    print(f"\nModel Configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Training accuracy: {checkpoint['accuracy']*100:.2f}%")
    print(f"  Training F1-score: {checkpoint['f1_score']:.4f}")
    
    # Initialize model
    model = StackedLSTM(
        input_size=63,
        hidden_size=config['hidden_size'],
        num_classes=NUM_CLASSES,
        num_lstm_layers=num_layers,
        dropout=config['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    print("\nLoading validation data...")
    X = np.load("outputs/sequences_X.npy")
    y = np.load("outputs/sequences_y.npy")
    video_ids = np.load("outputs/video_ids.npy", allow_pickle=True)
    
    # Use same split
    from sklearn.model_selection import train_test_split
    np.random.seed(RANDOM_SEED)
    unique_videos = np.unique(video_ids)
    train_videos, val_videos = train_test_split(
        unique_videos, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )
    
    val_mask = np.isin(video_ids, val_videos)
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    print(f"Validation sequences: {len(X_val)}")
    
    # Make predictions
    print("Making predictions...")
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    
    predictions = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(X_val), batch_size):
            batch = X_val_tensor[i:i+batch_size]
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_val, predictions, average='weighted', zero_division=0
    )
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"\nMacro-Averaged Metrics:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1-Score:  {f1_macro:.4f}")
    print(f"\nWeighted Metrics:")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall:    {recall_weighted:.4f}")
    print(f"  F1-Score:  {f1_weighted:.4f}")
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_val, predictions, 
                                       labels=range(NUM_CLASSES), zero_division=0)
    
    print(f"\n{'='*70}")
    print("PER-CLASS PERFORMANCE")
    print(f"{'='*70}")
    print(f"\n{'Class':<35} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 85)
    
    for class_id in range(NUM_CLASSES):
        if support_per_class[class_id] > 0:
            class_name = WHO_STEPS[class_id].split(":")[0] if ":" in WHO_STEPS[class_id] else WHO_STEPS[class_id]
            print(f"{class_name:<35} {precision_per_class[class_id]:<12.4f} "
                  f"{recall_per_class[class_id]:<12.4f} {f1_per_class[class_id]:<12.4f} "
                  f"{int(support_per_class[class_id])}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, predictions, labels=range(NUM_CLASSES))
    
    return {
        'num_layers': num_layers,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class_precision': precision_per_class,
        'per_class_recall': recall_per_class,
        'per_class_f1': f1_per_class,
        'per_class_support': support_per_class,
        'confusion_matrix': cm,
        'predictions': predictions,
        'ground_truth': y_val
    }

# ============================================================
# EVALUATE ALL CONFIGURATIONS
# ============================================================

def main():
    print("="*70)
    print("STACKED LSTM EVALUATION - ALL CONFIGURATIONS")
    print("="*70)
    
    all_results = {}
    
    for num_layers in [2, 3, 4, 5]:
        results = evaluate_stacked_model(num_layers)
        if results:
            all_results[f'{num_layers}-layer'] = results
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - ALL CONFIGURATIONS")
    print("="*70)
    
    print(f"\n{'Configuration':<20} {'Accuracy':<15} {'F1 (Weighted)':<15} {'F1 (Macro)'}")
    print("-" * 70)
    for config_name, results in all_results.items():
        print(f"{config_name:<20} {results['accuracy']*100:<14.2f}% "
              f"{results['f1_weighted']:<15.4f} {results['f1_macro']:.4f}")
    
    # Save results
    np.save(CHECKPOINT_DIR / "evaluation_results.npy", all_results)
    print(f"\n✓ Results saved to: {CHECKPOINT_DIR / 'evaluation_results.npy'}")

if __name__ == "__main__":
    main()