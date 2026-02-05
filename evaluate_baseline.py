"""
Evaluate Baseline LSTM Model
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
RANDOM_SEED = 42  # Must match training seed for same split

# WHO Step Names
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
# LOAD MODEL
# ============================================================

def load_baseline_model():
    """Load trained baseline model"""
    
    print("\n" + "="*70)
    print("LOADING BASELINE MODEL")
    print("="*70)
    
    # Load training history to get hyperparameters
    history_path = CHECKPOINT_DIR / "training_history.npy"
    if not history_path.exists():
        print(f"Error: Training history not found at {history_path}")
        print("Please train the model first using train_baseline.py")
        sys.exit(1)
    
    history = np.load(history_path, allow_pickle=True).item()
    hyperparams = history['hyperparameters']
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {hyperparams['hidden_size']}")
    print(f"  Learning rate: {hyperparams['learning_rate']}")
    print(f"  Dropout: {hyperparams['dropout']}")
    print(f"  Best training accuracy: {history['best_val_acc']:.2f}%")
    
    # Initialize model
    model = BaselineLSTM(
        input_size=63,  # 21 landmarks × 3 coordinates
        hidden_size=hyperparams['hidden_size'],
        num_classes=NUM_CLASSES,
        num_layers=2,
        dropout=hyperparams['dropout']
    ).to(DEVICE)
    
    # Load weights
    checkpoint_path = CHECKPOINT_DIR / "baseline_model.pt"
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_baseline.py")
        sys.exit(1)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    
    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_baseline():
    """Comprehensive evaluation of baseline model"""
    
    print("\n" + "="*70)
    print("BASELINE MODEL EVALUATION")
    print("="*70)
    
    # Load model
    model = load_baseline_model()
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X = np.load("outputs/sequences_X.npy")
    y = np.load("outputs/sequences_y.npy")
    video_ids = np.load("outputs/video_ids.npy", allow_pickle=True)
    
    # Use same split as training
    from sklearn.model_selection import train_test_split
    
    np.random.seed(RANDOM_SEED)
    unique_videos = np.unique(video_ids)
    train_videos, val_videos = train_test_split(
        unique_videos, 
        test_size=0.2, 
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Get validation data
    val_mask = np.isin(video_ids, val_videos)
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    print(f"Validation sequences: {len(X_val)}")
    print(f"Validation videos: {len(val_videos)}")
    
    # Make predictions
    print("\nMaking predictions...")
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(X_val), batch_size):
            batch = X_val_tensor[i:i+batch_size]
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    # Overall accuracy
    accuracy = accuracy_score(y_val, predictions)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, predictions, average='macro', zero_division=0
    )
    
    # Weighted metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_val, predictions, average='weighted', zero_division=0
    )
    
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
            if len(class_name) > 30:
                class_name = class_name[:27] + "..."
            
            print(f"{class_name:<35} {precision_per_class[class_id]:<12.4f} "
                  f"{recall_per_class[class_id]:<12.4f} {f1_per_class[class_id]:<12.4f} "
                  f"{int(support_per_class[class_id])}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, predictions, labels=range(NUM_CLASSES))
    
    print(f"\n{'='*70}")
    print("CONFUSION MATRIX ANALYSIS")
    print(f"{'='*70}")
    
    correct = cm.diagonal().sum()
    total = cm.sum()
    print(f"\nCorrect predictions: {correct}/{total} ({accuracy*100:.2f}%)")
    print(f"Misclassifications:  {total - correct}")
    
    # Top misclassifications
    print(f"\nTop 5 Misclassifications:")
    misclass = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                misclass.append((cm[i, j], i, j))
    
    misclass.sort(reverse=True)
    for count, true_class, pred_class in misclass[:5]:
        true_name = WHO_STEPS[true_class].split(":")[0]
        pred_name = WHO_STEPS[pred_class].split(":")[0]
        print(f"  {true_name} -> {pred_name}: {count} times")
    
    # Average confidence
    avg_confidence = probabilities.max(axis=1).mean() * 100
    print(f"\nAverage Prediction Confidence: {avg_confidence:.1f}%")
    
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
        'avg_confidence': avg_confidence,
        'predictions': predictions,
        'ground_truth': y_val
    }
    
    results_path = CHECKPOINT_DIR / "evaluation_results.npy"
    np.save(results_path, results)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Generate visualizations
    generate_visualizations(cm, results)
    
    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def generate_visualizations(cm, results):
    """Generate confusion matrix and per-class metrics plots"""
    
    print("\nGenerating visualizations...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    
    labels = [WHO_STEPS[i].split(":")[0] if ":" in WHO_STEPS[i] else WHO_STEPS[i] 
              for i in range(NUM_CLASSES)]
    labels = [label[:15] + "..." if len(label) > 15 else label for label in labels]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Baseline LSTM Model', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = CHECKPOINT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Confusion matrix saved to: {cm_path}")
    plt.close()
    
    # 2. Per-class metrics bar chart
    valid_classes = [i for i in range(NUM_CLASSES) if results['per_class_support'][i] > 0]
    class_names = [WHO_STEPS[i].split(":")[0] for i in valid_classes]
    
    precision = [results['per_class_precision'][i] for i in valid_classes]
    recall = [results['per_class_recall'][i] for i in valid_classes]
    f1 = [results['per_class_f1'][i] for i in valid_classes]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('WHO Step', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics - Baseline LSTM', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    metrics_path = CHECKPOINT_DIR / "per_class_metrics.png"
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Per-class metrics saved to: {metrics_path}")
    plt.close()


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(results):
    """Generate detailed text report"""
    
    report_path = CHECKPOINT_DIR / "evaluation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BASELINE LSTM MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("Model: Single Baseline LSTM with Attention\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Device: {DEVICE}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"Accuracy:             {results['accuracy']*100:.2f}%\n")
        f.write(f"Precision (Macro):    {results['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro):       {results['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro):     {results['f1_macro']:.4f}\n")
        f.write(f"Precision (Weighted): {results['precision_weighted']:.4f}\n")
        f.write(f"Recall (Weighted):    {results['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Weighted):  {results['f1_weighted']:.4f}\n")
        f.write(f"Avg Confidence:       {results['avg_confidence']:.1f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"{'Class':<35} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}\n")
        f.write("-"*85 + "\n")
        
        for class_id in range(NUM_CLASSES):
            if results['per_class_support'][class_id] > 0:
                class_name = WHO_STEPS[class_id].split(":")[0]
                f.write(f"{class_name:<35} "
                       f"{results['per_class_precision'][class_id]:<12.4f} "
                       f"{results['per_class_recall'][class_id]:<12.4f} "
                       f"{results['per_class_f1'][class_id]:<12.4f} "
                       f"{int(results['per_class_support'][class_id])}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"  ✓ Detailed report saved to: {report_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("BASELINE LSTM MODEL EVALUATION")
    print("="*70)
    
    try:
        results = evaluate_baseline()
        generate_report(results)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"\nFinal Accuracy: {results['accuracy']*100:.2f}%")
        print(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"\nAll results saved to: {CHECKPOINT_DIR}")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)