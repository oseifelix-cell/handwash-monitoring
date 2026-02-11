"""
Calculate Overall Accuracy for 8-Model Ensemble
Handles missing models gracefully (like Model 8 which may have failed).
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.specialized_lstm import SpecializedLSTM

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("outputs/checkpoints_8model_ensemble")
RANDOM_SEED = 42

print("="*70)
print("8-MODEL ENSEMBLE - OVERALL ACCURACY CALCULATION")
print("="*70)

# Load data
print("\nLoading data...")
X = np.load("outputs/sequences_X.npy")
y = np.load("outputs/sequences_y.npy")
video_ids = np.load("outputs/video_ids.npy", allow_pickle=True)

# Get validation split (using Model 1's seed)
from sklearn.model_selection import train_test_split
np.random.seed(RANDOM_SEED + 1)
unique_videos = np.unique(video_ids)
train_videos, val_videos = train_test_split(
    unique_videos, test_size=0.2, random_state=RANDOM_SEED + 1, shuffle=True
)

val_mask = np.isin(video_ids, val_videos)
X_val = X[val_mask]
y_val = y[val_mask]

print(f"Validation sequences: {len(X_val)}")

# Load all available models
print("\nLoading models...")
models = []
model_info = []
available_steps = []

for model_id in range(1, 9):
    checkpoint_path = CHECKPOINT_DIR / f"model_{model_id}.pt"
    
    if not checkpoint_path.exists():
        print(f"  ⚠️ Model {model_id} not found - skipping")
        continue
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Check if model actually learned something
    if checkpoint.get('best_f1', 0) == 0:
        print(f"  ⚠️ Model {model_id} (Step {checkpoint['target_step']}) failed training (F1=0.0000) - skipping")
        continue
    
    model = SpecializedLSTM(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        dropout=0.4
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    models.append(model)
    model_info.append({
        'model_id': model_id,
        'target_step': checkpoint['target_step'],
        'step_name': checkpoint['step_name'],
        'f1': checkpoint['best_f1']
    })
    available_steps.append(checkpoint['target_step'])
    
    print(f"  ✓ Model {model_id} ({checkpoint['step_name']}): F1={checkpoint['best_f1']:.4f}")

if len(models) == 0:
    print("\nError: No valid models found!")
    sys.exit(1)

print(f"\nUsing {len(models)} models (skipped {8 - len(models)} failed/missing models)")

# Make predictions
print("\nMaking predictions...")
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)

# Get probability for each available step
step_probabilities = np.zeros((len(X_val), len(models)))

with torch.no_grad():
    batch_size = 64
    for model_idx, model in enumerate(models):
        probs_for_step = []
        
        for i in range(0, len(X_val), batch_size):
            batch = X_val_tensor[i:i+batch_size]
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            # Probability that it IS the target step (class 1)
            probs_for_step.append(probs[:, 1].cpu().numpy())
        
        step_probabilities[:, model_idx] = np.concatenate(probs_for_step)

# Make final predictions
print("\nCombining predictions...")
predictions = []

for i in range(len(X_val)):
    # Get the step with highest probability
    max_prob = step_probabilities[i].max()
    best_model_idx = step_probabilities[i].argmax()
    predicted_step = available_steps[best_model_idx]
    
    # If confidence is low, predict background (0)
    if max_prob < 0.3:  # Lower threshold
        predictions.append(0)
    else:
        predictions.append(predicted_step)

predictions = np.array(predictions)

# Calculate overall accuracy
print("\n" + "="*70)
print("OVERALL SYSTEM PERFORMANCE")
print("="*70)

accuracy = accuracy_score(y_val, predictions)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_val, predictions, average='macro', zero_division=0
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_val, predictions, average='weighted', zero_division=0
)

print(f"\n🎯 ENSEMBLE ACCURACY: {accuracy*100:.2f}%")
print(f"   (Using {len(models)} out of 8 models)")
print(f"\nMacro-Averaged Metrics:")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall:    {recall_macro:.4f}")
print(f"  F1-Score:  {f1_macro:.4f}")

print(f"\nWeighted Metrics:")
print(f"  Precision: {precision_weighted:.4f}")
print(f"  Recall:    {recall_weighted:.4f}")
print(f"  F1-Score:  {f1_weighted:.4f}")

# Per-class breakdown
precision_per_class, recall_per_class, f1_per_class, support_per_class = \
    precision_recall_fscore_support(y_val, predictions, labels=range(9), zero_division=0)

print(f"\n{'='*70}")
print("PER-CLASS PERFORMANCE")
print(f"{'='*70}")
print(f"\n{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
print("-" * 75)

WHO_STEPS = ["Background", "Step 1", "Step 2", "Step 3", "Step 4", 
             "Step 5", "Step 6", "Step 7", "Step 8"]

for class_id in range(9):
    if support_per_class[class_id] > 0:
        marker = " ⚠️" if class_id not in available_steps and class_id != 0 else ""
        print(f"{WHO_STEPS[class_id]:<25} {precision_per_class[class_id]:<12.4f} "
              f"{recall_per_class[class_id]:<12.4f} {f1_per_class[class_id]:<12.4f} "
              f"{int(support_per_class[class_id])}{marker}")

if 8 not in available_steps:
    print("\n⚠️ = Model not available (failed training or missing)")

# Confusion matrix analysis
cm = confusion_matrix(y_val, predictions, labels=range(9))
correct = cm.diagonal().sum()
total = cm.sum()

print(f"\n{'='*70}")
print("CONFUSION MATRIX ANALYSIS")
print(f"{'='*70}")
print(f"\nCorrect predictions: {correct}/{total} ({accuracy*100:.2f}%)")
print(f"Misclassifications:  {total - correct}")

# Save results
results = {
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
    'ground_truth': y_val,
    'num_models_used': len(models),
    'available_steps': available_steps,
    'model_info': model_info
}

np.save(CHECKPOINT_DIR / "ensemble_overall_accuracy.npy", results)
print(f"\n✓ Results saved to: {CHECKPOINT_DIR / 'ensemble_overall_accuracy.npy'}")