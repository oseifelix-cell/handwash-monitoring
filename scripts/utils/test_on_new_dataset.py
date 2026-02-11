"""
Test Ensemble Model on New Annotated Dataset
This script processes a completely new dataset with annotations and evaluates the ensemble model.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.feature_extractor import extract_features, SEQ_LEN
from src.models.lstm_model import HandwashLSTM
from src.utils.annotation_loader import load_annotations, get_label_for_sequence

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("outputs/checkpoints_ensemble")
NUM_CLASSES = 9

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

# Model configurations (same as training)
CONFIGS = [
    {"seed": 42, "hidden": 128, "dropout": 0.3},
    {"seed": 123, "hidden": 112, "dropout": 0.4},
    {"seed": 777, "hidden": 144, "dropout": 0.35},
    {"seed": 2024, "hidden": 128, "dropout": 0.45},
    {"seed": 999, "hidden": 96, "dropout": 0.5},
]

# ============================================================
# LOAD ENSEMBLE MODELS
# ============================================================

def load_ensemble():
    """Load all 5 trained models"""
    print("\n" + "="*70)
    print("LOADING ENSEMBLE MODELS")
    print("="*70)
    
    models = []
    
    for i, config in enumerate(CONFIGS, 1):
        model = HandwashLSTM(
            input_size=63,
            hidden_size=config["hidden"],
            num_classes=NUM_CLASSES,
            num_layers=2,
            dropout=config["dropout"]
        ).to(DEVICE)
        
        checkpoint_path = CHECKPOINT_DIR / f"model_{i}.pt"
        
        if not checkpoint_path.exists():
            print(f"Error: {checkpoint_path} not found!")
            print(f"Please ensure you've trained the ensemble models first.")
            sys.exit(1)
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        models.append(model)
        print(f"  Model {i} loaded successfully")
    
    print("\nAll 5 models loaded!")
    return models

# ============================================================
# PROCESS NEW DATASET
# ============================================================

def process_new_dataset(video_dir, annotation_dir, models):
    """
    Process all videos in the new dataset and get predictions with ground truth.
    
    Args:
        video_dir: Path to directory containing new videos
        annotation_dir: Path to directory containing annotations
        models: List of loaded ensemble models
    
    Returns:
        all_predictions: Array of predicted labels
        all_ground_truth: Array of ground truth labels
        all_confidences: Array of confidence scores
        video_results: Dictionary with per-video results
    """
    print("\n" + "="*70)
    print("PROCESSING NEW DATASET")
    print("="*70)
    
    video_dir = Path(video_dir)
    annotation_dir = Path(annotation_dir)
    
    # Get all video files
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"Error: No .mp4 files found in {video_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(video_files)} videos in {video_dir}")
    
    # Load annotations
    print(f"\nLoading annotations from {annotation_dir}...")
    annotations = load_annotations(annotation_dir)
    print(f"Loaded annotations for {len(annotations)} videos")
    
    # Process each video
    all_predictions = []
    all_ground_truth = []
    all_confidences = []
    video_results = {}
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = video_path.stem
        
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_name}")
        
        # Check if annotations exist for this video
        if video_name not in annotations:
            print(f"  Warning: No annotations found for {video_name}, skipping...")
            continue
        
        # Extract features
        result = extract_features(str(video_path), return_frames=True)
        
        if result is None:
            print(f"  Warning: Could not extract features from {video_name}, skipping...")
            continue
        
        sequences, seq_start_frames = result
        print(f"  Extracted {len(sequences)} sequences")
        
        # Get ground truth labels for each sequence
        frame_labels = annotations[video_name]
        sequence_labels = []
        
        for start_frame in seq_start_frames:
            label = get_label_for_sequence(start_frame, SEQ_LEN, frame_labels)
            if label is not None and label >= 0:
                sequence_labels.append(label)
            else:
                sequence_labels.append(-1)  # Invalid label
        
        # Filter out sequences with invalid labels
        valid_indices = [i for i, label in enumerate(sequence_labels) if label >= 0]
        
        if len(valid_indices) == 0:
            print(f"  Warning: No valid labeled sequences found for {video_name}, skipping...")
            continue
        
        sequences = sequences[valid_indices]
        sequence_labels = [sequence_labels[i] for i in valid_indices]
        
        print(f"  Valid sequences with labels: {len(sequence_labels)}")
        
        # Convert to tensor
        X = torch.tensor(sequences, dtype=torch.float32).to(DEVICE)
        
        # Get predictions from all models
        model_predictions = []
        
        with torch.no_grad():
            for model in models:
                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)
                model_predictions.append(probs.cpu().numpy())
        
        # Average predictions (ensemble)
        avg_predictions = np.mean(model_predictions, axis=0)
        predictions = avg_predictions.argmax(axis=1)
        confidences = avg_predictions.max(axis=1)
        
        # Calculate accuracy for this video
        video_accuracy = accuracy_score(sequence_labels, predictions)
        
        print(f"  Video Accuracy: {video_accuracy*100:.2f}%")
        
        # Store results
        all_predictions.extend(predictions)
        all_ground_truth.extend(sequence_labels)
        all_confidences.extend(confidences)
        
        video_results[video_name] = {
            'predictions': predictions,
            'ground_truth': np.array(sequence_labels),
            'confidences': confidences,
            'accuracy': video_accuracy,
            'num_sequences': len(predictions)
        }
    
    print(f"\n" + "="*70)
    print(f"Dataset processing complete!")
    print(f"Total videos processed: {len(video_results)}")
    print(f"Total sequences: {len(all_predictions)}")
    print("="*70)
    
    return (np.array(all_predictions), 
            np.array(all_ground_truth), 
            np.array(all_confidences),
            video_results)

# ============================================================
# CALCULATE COMPREHENSIVE METRICS
# ============================================================

def calculate_metrics(predictions, ground_truth, confidences):
    """Calculate all performance metrics"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE PERFORMANCE METRICS")
    print("="*70)
    
    # Overall metrics
    accuracy = accuracy_score(ground_truth, predictions)
    
    # Macro-averaged metrics (equal weight to each class)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='macro', zero_division=0
    )
    
    # Weighted metrics (weight by class frequency)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted', zero_division=0
    )
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"{'Metric':<25} {'Value'}")
    print("-" * 45)
    print(f"{'Accuracy':<25} {accuracy*100:.2f}%")
    print(f"{'Precision (Macro)':<25} {precision_macro:.4f}")
    print(f"{'Recall (Macro)':<25} {recall_macro:.4f}")
    print(f"{'F1-Score (Macro)':<25} {f1_macro:.4f}")
    print(f"{'Precision (Weighted)':<25} {precision_weighted:.4f}")
    print(f"{'Recall (Weighted)':<25} {recall_weighted:.4f}")
    print(f"{'F1-Score (Weighted)':<25} {f1_weighted:.4f}")
    print(f"{'Avg Confidence':<25} {confidences.mean()*100:.1f}%")
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(ground_truth, predictions, 
                                       labels=range(NUM_CLASSES), zero_division=0)
    
    print(f"\nPER-CLASS PERFORMANCE:")
    print(f"{'Class':<35} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
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
    cm = confusion_matrix(ground_truth, predictions, labels=range(NUM_CLASSES))
    
    print(f"\nCONFUSION MATRIX SUMMARY:")
    correct = cm.diagonal().sum()
    total = cm.sum()
    print(f"  Correct predictions: {correct}/{total} ({accuracy*100:.2f}%)")
    print(f"  Misclassifications:  {total - correct}")
    
    # Top misclassifications
    print(f"\nTOP 5 MISCLASSIFICATIONS:")
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
    
    # Class distribution
    print(f"\nCLASS DISTRIBUTION IN DATASET:")
    class_counts = Counter(ground_truth)
    print(f"{'Class':<35} {'Count':<10} {'Percentage'}")
    print("-" * 60)
    for class_id in sorted(class_counts.keys()):
        class_name = WHO_STEPS[class_id].split(":")[0]
        count = class_counts[class_id]
        percentage = (count / len(ground_truth)) * 100
        print(f"{class_name:<35} {count:<10} {percentage:.2f}%")
    
    metrics = {
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
        'confusion_matrix': cm
    }
    
    return metrics

# ============================================================
# VISUALIZATIONS
# ============================================================

def plot_confusion_matrix(cm, save_path="new_dataset_confusion_matrix.png"):
    """Plot confusion matrix heatmap"""
    
    print(f"\nGenerating confusion matrix visualization...")
    
    plt.figure(figsize=(12, 10))
    
    # Create labels
    labels = [WHO_STEPS[i].split(":")[0] if ":" in WHO_STEPS[i] else WHO_STEPS[i] 
              for i in range(NUM_CLASSES)]
    
    # Shorten labels if too long
    labels = [label[:15] + "..." if len(label) > 15 else label for label in labels]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - New Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")

def plot_per_class_metrics(metrics, save_path="new_dataset_per_class_metrics.png"):
    """Plot per-class precision, recall, F1-score"""
    
    print(f"\nGenerating per-class metrics visualization...")
    
    # Filter out classes with no support
    valid_classes = [i for i in range(NUM_CLASSES) if metrics['per_class_support'][i] > 0]
    
    class_names = [WHO_STEPS[i].split(":")[0] for i in valid_classes]
    precision = [metrics['per_class_precision'][i] for i in valid_classes]
    recall = [metrics['per_class_recall'][i] for i in valid_classes]
    f1 = [metrics['per_class_f1'][i] for i in valid_classes]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('WHO Step', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics - New Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics plot saved to: {save_path}")

# ============================================================
# SAVE RESULTS
# ============================================================

def save_detailed_report(predictions, ground_truth, confidences, metrics, 
                        video_results, save_path="new_dataset_results.txt"):
    """Save comprehensive results to text file"""
    
    print(f"\nSaving detailed report...")
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ENSEMBLE MODEL EVALUATION ON NEW DATASET\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System: Ensemble LSTM (5 models)\n")
        f.write(f"Total Videos: {len(video_results)}\n")
        f.write(f"Total Sequences: {len(predictions)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"Accuracy:             {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision (Macro):    {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro):       {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro):     {metrics['f1_macro']:.4f}\n")
        f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Weighted):  {metrics['f1_weighted']:.4f}\n")
        f.write(f"Average Confidence:   {confidences.mean()*100:.1f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"{'Class':<35} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}\n")
        f.write("-"*85 + "\n")
        
        for class_id in range(NUM_CLASSES):
            if metrics['per_class_support'][class_id] > 0:
                class_name = WHO_STEPS[class_id].split(":")[0]
                f.write(f"{class_name:<35} "
                       f"{metrics['per_class_precision'][class_id]:<12.4f} "
                       f"{metrics['per_class_recall'][class_id]:<12.4f} "
                       f"{metrics['per_class_f1'][class_id]:<12.4f} "
                       f"{int(metrics['per_class_support'][class_id])}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("PER-VIDEO RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"{'Video Name':<40} {'Accuracy':<12} {'Sequences'}\n")
        f.write("-"*60 + "\n")
        
        for video_name, result in sorted(video_results.items()):
            f.write(f"{video_name:<40} {result['accuracy']*100:<12.2f} {result['num_sequences']}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"Detailed report saved to: {save_path}")

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("="*70)
    print("ENSEMBLE MODEL TESTING ON NEW ANNOTATED DATASET")
    print("="*70)
    
    # Get paths from user
    if len(sys.argv) >= 3:
        video_dir = sys.argv[1]
        annotation_dir = sys.argv[2]
    else:
        print("\nPlease provide paths to your new dataset:")
        video_dir = input("Video directory path: ").strip().strip('"')
        annotation_dir = input("Annotation directory path: ").strip().strip('"')
    
    # Validate paths
    if not Path(video_dir).exists():
        print(f"Error: Video directory not found: {video_dir}")
        return
    
    if not Path(annotation_dir).exists():
        print(f"Error: Annotation directory not found: {annotation_dir}")
        return
    
    # Load ensemble models
    models = load_ensemble()
    
    # Process new dataset
    predictions, ground_truth, confidences, video_results = process_new_dataset(
        video_dir, annotation_dir, models
    )
    
    if len(predictions) == 0:
        print("\nError: No valid data processed. Please check your dataset.")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth, confidences)
    
    # Generate visualizations
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_per_class_metrics(metrics)
    
    # Save detailed report
    save_detailed_report(predictions, ground_truth, confidences, metrics, video_results)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Accuracy:   {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score:   {metrics['f1_weighted']:.4f}")
    print(f"  Precision:  {metrics['precision_weighted']:.4f}")
    print(f"  Recall:     {metrics['recall_weighted']:.4f}")
    print(f"\nFiles saved:")
    print(f"  - new_dataset_confusion_matrix.png")
    print(f"  - new_dataset_per_class_metrics.png")
    print(f"  - new_dataset_results.txt")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()