"""
Comprehensive evaluation of ensemble model performance.
Generates confusion matrix, per-class metrics, and visualizations.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.lstm_model import HandwashLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("outputs/checkpoints_ensemble")

# Load data
X = np.load("outputs/sequences_X.npy")
y = np.load("outputs/sequences_y.npy")
video_ids = np.load("outputs/video_ids.npy")

NUM_CLASSES = len(np.unique(y))

# Use same validation split as training
unique_videos = np.unique(video_ids)
_, val_videos = train_test_split(unique_videos, test_size=0.2, random_state=42, shuffle=True)
val_mask = np.isin(video_ids, val_videos)

X_val = torch.tensor(X[val_mask], dtype=torch.float32).to(DEVICE)
y_val = y[val_mask]

print("=" * 70)
print("🔍 ENSEMBLE MODEL EVALUATION")
print("=" * 70)
print(f"Validation sequences: {len(y_val)}")
print(f"Validation videos: {len(val_videos)}")

# Model configurations
configs = [
    (42, 128, 1e-3, 0.3),
    (123, 112, 8e-4, 0.4),
    (777, 144, 1.2e-3, 0.35),
    (2024, 128, 7e-4, 0.45),
    (999, 96, 9e-4, 0.5),
]

# Load all models and get predictions
print("\n📥 Loading ensemble models...")
all_predictions = []
individual_accs = []

for i, (seed, hidden, lr, dropout) in enumerate(configs, 1):
    model = HandwashLSTM(
        input_size=X.shape[2],
        hidden_size=hidden,
        num_classes=NUM_CLASSES,
        num_layers=2,
        dropout=dropout
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"model_{i}.pt"))
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_val)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu().numpy()
        
        # Individual accuracy
        acc = 100 * (preds == y_val).sum() / len(y_val)
        individual_accs.append(acc)
        print(f"  Model {i}: {acc:.2f}% accuracy")
        
        all_predictions.append(probs.cpu().numpy())

# Ensemble predictions
avg_predictions = np.mean(all_predictions, axis=0)
ensemble_preds = avg_predictions.argmax(axis=1)
ensemble_acc = 100 * (ensemble_preds == y_val).sum() / len(y_val)

print(f"\n🏆 Ensemble Accuracy: {ensemble_acc:.2f}%")
print(f"   Average individual: {np.mean(individual_accs):.2f}%")
print(f"   Ensemble boost: +{ensemble_acc - np.mean(individual_accs):.2f}%")

# WHO step names
class_names = [
    "Background/No Washing",
    "Step 1: Palm to Palm",
    "Step 2: Right Palm over Left Dorsum",
    "Step 3: Left Palm over Right Dorsum",
    "Step 4: Fingers Interlaced",
    "Step 5: Backs of Fingers",
    "Step 6: Rotational Rubbing of Thumbs",
    "Step 7: Rotational Rubbing of Fingertips",
    "Step 8: Rotational Rubbing of Wrists"
]

# Ensure we only use names for classes in data
unique_classes = np.unique(y_val)
used_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]

# Classification report
print("\n" + "=" * 70)
print("📊 DETAILED CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_val,
    ensemble_preds,
    labels=unique_classes,
    target_names=used_class_names,
    digits=3
)
print(report)

# Per-class accuracy
print("\n" + "=" * 70)
print("🎯 PER-CLASS PERFORMANCE")
print("=" * 70)

per_class_correct = []
per_class_total = []

for cls in unique_classes:
    mask = y_val == cls
    correct = (ensemble_preds[mask] == y_val[mask]).sum()
    total = mask.sum()
    acc = 100 * correct / total if total > 0 else 0
    
    per_class_correct.append(correct)
    per_class_total.append(total)
    
    class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
    print(f"{class_name:45s}: {acc:6.2f}% ({correct:4d}/{total:4d})")

# Macro and Weighted F1
macro_f1 = f1_score(y_val, ensemble_preds, average='macro')
weighted_f1 = f1_score(y_val, ensemble_preds, average='weighted')

print(f"\n📈 Overall Metrics:")
print(f"  Macro F1-Score:    {macro_f1:.3f}")
print(f"  Weighted F1-Score: {weighted_f1:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_val, ensemble_preds, labels=unique_classes)

# Normalize confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Raw counts
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=used_class_names,
    yticklabels=used_class_names,
    ax=ax1,
    cbar_kws={'label': 'Count'}
)
ax1.set_title('Confusion Matrix - Raw Counts', fontsize=14, pad=20)
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax1.get_yticklabels(), rotation=0)

# Normalized percentages
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2%',
    cmap='Blues',
    xticklabels=used_class_names,
    yticklabels=used_class_names,
    ax=ax2,
    cbar_kws={'label': 'Percentage'}
)
ax2.set_title('Confusion Matrix - Normalized', fontsize=14, pad=20)
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xlabel('Predicted Label', fontsize=12)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax2.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('outputs/ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Confusion matrix saved to outputs/ensemble_confusion_matrix.png")

# Per-class accuracy bar chart
fig, ax = plt.subplots(figsize=(12, 8))

per_class_accs = [100 * per_class_correct[i] / per_class_total[i] 
                  for i in range(len(unique_classes))]

bars = ax.barh(used_class_names, per_class_accs, color='steelblue', edgecolor='black')

# Color code: green if >85%, yellow if 70-85%, red if <70%
for i, (bar, acc) in enumerate(zip(bars, per_class_accs)):
    if acc >= 85:
        bar.set_color('green')
    elif acc >= 70:
        bar.set_color('gold')
    else:
        bar.set_color('tomato')
    
    # Add percentage label
    ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('Per-Class Accuracy - Ensemble Model', fontsize=14, pad=20)
ax.set_xlim(0, 105)
ax.axvline(70, color='red', linestyle='--', alpha=0.3, label='70% threshold')
ax.axvline(85, color='green', linestyle='--', alpha=0.3, label='85% threshold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("✅ Per-class accuracy chart saved to outputs/per_class_accuracy.png")

# Model contribution analysis
print("\n" + "=" * 70)
print("🤝 ENSEMBLE MODEL CONTRIBUTIONS")
print("=" * 70)

# See which models contribute most to correct predictions
model_contributions = np.zeros(5)

for i in range(len(y_val)):
    true_label = y_val[i]
    
    # Check which individual models got it right
    for model_idx in range(5):
        if all_predictions[model_idx][i].argmax() == true_label:
            model_contributions[model_idx] += 1

for i, contrib in enumerate(model_contributions, 1):
    percentage = 100 * contrib / len(y_val)
    print(f"  Model {i}: Correct on {contrib:4.0f}/{len(y_val)} = {percentage:.1f}%")

# Agreement analysis
all_model_preds = np.array([[pred[i].argmax() for pred in all_predictions] 
                            for i in range(len(y_val))])

unanimous_correct = 0
majority_correct = 0

for i in range(len(y_val)):
    preds = all_model_preds[i]
    true_label = y_val[i]
    
    if np.all(preds == true_label):
        unanimous_correct += 1
    elif np.sum(preds == true_label) >= 3:
        majority_correct += 1

print(f"\n📊 Agreement Statistics:")
print(f"  Unanimous correct: {unanimous_correct} ({100*unanimous_correct/len(y_val):.1f}%)")
print(f"  Majority correct:  {majority_correct} ({100*majority_correct/len(y_val):.1f}%)")
print(f"  Disagreement:      {len(y_val)-unanimous_correct-majority_correct}")

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETE!")
print("=" * 70)
print(f"\n🎓 FINAL RESULT: {ensemble_acc:.2f}% Validation Accuracy")
print(f"   Baseline (original): 43.00%")
print(f"   Improvement: +{ensemble_acc - 43:.2f} percentage points")