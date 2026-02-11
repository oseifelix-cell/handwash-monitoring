"""
Evaluate 8-Model Specialized Ensemble
Combines predictions from 8 expert models to make final WHO step classification.
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

from src.models.specialized_lstm import SpecializedLSTM

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 9
CHECKPOINT_DIR = Path("outputs/checkpoints_8model_ensemble")
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

# ============================================================
# LOAD MODELS
# ============================================================

def load_8model_ensemble():
    """Load all 8 specialized models"""
    
    print("\n" + "="*70)
    print("LOADING 8-MODEL ENSEMBLE")
    print("="*70)
    
    models = []
    model_info = []
    
    for model_id in range(1, 9):
        checkpoint_path = CHECKPOINT_DIR / f"model_{model_id}.pt"
        
        if not checkpoint_path.exists():
            print(f"Error: {checkpoint_path} not found!")
            print("Please train models first using train_8model_ensemble.py")
            sys.exit(1)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Initialize model
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
            'target_step': checkpoint['target_step'],
            'step_name': checkpoint['step_name'],
            'best_f1': checkpoint['best_f1'],
            'accuracy': checkpoint['accuracy']
        })
        
        print(f"  ✓ Model {model_id} ({checkpoint['step_name']}): F1={checkpoint['best_f1']:.4f}")
    
    print("\n✓ All 8 models loaded successfully!")
    return models, model_info

# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(models, model_info, X):
    """
    Combine predictions from 8 specialized models.
    
    Strategy:
    1. Each model predicts probability that sequence is its target step
    2. Combine probabilities to form 9-class prediction (including background)
    3. Background is predicted when no specific step has high confidence
    """
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    batch_size = 64
    
    # Get predictions from each model
    all_step_probs = np.zeros((len(X), 8))  # Probabilities for steps 1-8
    
    with torch.no_grad():
        for model_idx, (model, info) in enumerate(zip(models, model_info)):
            step_probs_batch = []
            
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i:i+batch_size]
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
                # Get probability of being the target step (class 1)
                step_probs_batch.append(probs[:, 1].cpu().numpy())
            
            all_step_probs[:, model_idx] = np.concatenate(step_probs_batch)
    
    # Make final predictions
    predictions = []
    confidences = []
    
    for i in range(len(X)):
        # Get max probability across all 8 models
        max_prob = all_step_probs[i].max()
        max_step = all_step_probs[i].argmax() + 1  # +1 because steps are 1-8
        
        # If confidence is low, predict background
        if max_prob < 0.5:  # Threshold for classifying as background
            predictions.append(0)  # Background
            confidences.append(1.0 - max_prob)  # Confidence in background
        else:
            predictions.append(max_step)
            confidences.append(max_prob)
    
    return np.array(predictions), np.array(confidences)

# ============================================================
# EVALUATE ENSEMBLE
# ============================================================

def evaluate_ensemble():
    """Comprehensive evaluation of 8-model ensemble"""
    
    print("\n" + "="*70)
    print("8-MODEL ENSEMBLE EVALUATION")
    print("="*70)
    
    # Load models
    models, model_info = load_8model_ensemble()
    
    # Load data
    print("\nLoading validation data...")
    X = np.load("outputs/sequences_X.npy")
    y = np.load("outputs/sequences_y.npy")
    video_ids = np.load("outputs/video_ids.npy", allow_pickle=True)
    
    # Use same split as training (Model 1's split)
    from sklearn.model_selection import train_test_split
    
    np.random.seed(RANDOM_SEED + 1)  # Model 1's seed
    unique_videos = np.unique(video_ids)
    train_videos, val_videos = train_test_split(
        unique_videos, test_size=0.2, 
        random_state=RANDOM_SEED + 1, shuffle=True
    )
    
    val_mask = np.isin(video_ids, val_videos)
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    print(f"Validation sequences: {len(X_val)}")
    print(f"Validation videos: {len(val_videos)}")
    
    # Get ensemble predictions
    print("\nMaking ensemble predictions...")
    predictions, confidences = ensemble_predict(models, model_info, X_val)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    accuracy = accuracy_score(y_val, predictions)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    # Macro and weighted metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, predictions, average='macro', zero_division=0
    )
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
        'confidences': confidences
    }
    
    results_path = CHECKPOINT_DIR / "ensemble_evaluation_results.npy"
    np.save(results_path, results)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Generate visualizations
    generate_visualizations(cm, results)
    generate_report(results, model_info)
    
    return results

# ============================================================
# VISUALIZATIONS
# ============================================================

def generate_visualizations(cm, results):
    """Generate confusion matrix and metrics plots"""
    
    print("\nGenerating visualizations...")
    
    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    labels = [WHO_STEPS[i].split(":")[0] if ":" in WHO_STEPS[i] else WHO_STEPS[i] 
              for i in range(NUM_CLASSES)]
    labels = [label[:15] + "..." if len(label) > 15 else label for label in labels]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - 8-Model Specialized Ensemble', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(CHECKPOINT_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Confusion matrix saved")
    plt.close()
    
    # Per-class metrics
    valid_classes = [i for i in range(NUM_CLASSES) if results['per_class_support'][i] > 0]
    class_names = [WHO_STEPS[i].split(":")[0] for i in valid_classes]
    
    precision = [results['per_class_precision'][i] for i in valid_classes]
    recall = [results['per_class_recall'][i] for i in valid_classes]
    f1 = [results['per_class_f1'][i] for i in valid_classes]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('WHO Step', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance - 8-Model Ensemble', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "per_class_metrics.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Per-class metrics saved")
    plt.close()

# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(results, model_info):
    """Generate detailed text report"""
    
    report_path = CHECKPOINT_DIR / "evaluation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("8-MODEL SPECIALIZED ENSEMBLE EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("Architecture: 8 specialized binary classifiers\n")
        f.write("Each model trained to recognize one WHO step vs background\n\n")
        
        f.write("-"*70 + "\n")
        f.write("INDIVIDUAL MODEL PERFORMANCE\n")
        f.write("-"*70 + "\n\n")
        
        for i, info in enumerate(model_info, 1):
            f.write(f"Model {i} ({info['step_name']}):\n")
            f.write(f"  F1-Score: {info['best_f1']:.4f}\n")
            f.write(f"  Accuracy: {info['accuracy']*100:.2f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("ENSEMBLE PERFORMANCE\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"Accuracy:             {results['accuracy']*100:.2f}%\n")
        f.write(f"Precision (Macro):    {results['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro):       {results['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro):     {results['f1_macro']:.4f}\n")
        f.write(f"Precision (Weighted): {results['precision_weighted']:.4f}\n")
        f.write(f"Recall (Weighted):    {results['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Weighted):  {results['f1_weighted']:.4f}\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"  ✓ Detailed report saved")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("8-MODEL SPECIALIZED ENSEMBLE EVALUATION")
    print("="*70)
    
    try:
        results = evaluate_ensemble()
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"\nFinal Accuracy: {results['accuracy']*100:.2f}%")
        print(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)