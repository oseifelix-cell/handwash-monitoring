"""
Test New Handwashing Video
Analyzes a new video and determines if WHO handwashing steps were performed correctly.
"""
import pandas as pd
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.feature_extractor import extract_features, SEQ_LEN
from src.models.lstm_model import HandwashLSTM

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

# Required WHO steps for proper handwashing (1-7)
REQUIRED_STEPS = [1, 2, 3, 4, 5, 6, 7]

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
    print("📥 Loading ensemble models...")
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
            print(f"❌ Error: {checkpoint_path} not found!")
            print(f"   Please ensure you've trained the ensemble models first.")
            sys.exit(1)
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        models.append(model)
        print(f"  ✓ Model {i} loaded")
    
    print("✅ All models loaded successfully!\n")
    return models

# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(video_path, models):
    """
    Process a new video and predict WHO steps for each sequence.
    
    Returns:
        predictions: List of predicted WHO steps for each sequence
        confidences: List of confidence scores
        sequences: The extracted sequences
    """
    print(f"🎥 Processing video: {video_path}")
    
    # Extract features
    result = extract_features(video_path, return_frames=True)
    
    if result is None:
        print("❌ Error: Could not extract features from video.")
        print("   Possible reasons:")
        print("   - Video too short (need at least 30 frames with hands visible)")
        print("   - No hands detected")
        print("   - Video file corrupted")
        return None, None, None
    
    sequences, seq_start_frames = result
    print(f"✓ Extracted {len(sequences)} sequences from video")
    
    # Convert to tensor
    X = torch.tensor(sequences, dtype=torch.float32).to(DEVICE)
    
    # Get predictions from all models
    all_predictions = []
    
    with torch.no_grad():
        for model in models:
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            all_predictions.append(probs.cpu().numpy())
    
    # Average predictions (ensemble)
    avg_predictions = np.mean(all_predictions, axis=0)
    
    # Get predicted classes and confidence scores
    predictions = avg_predictions.argmax(axis=1)
    confidences = avg_predictions.max(axis=1)
    
    return predictions, confidences, seq_start_frames

# ============================================================
# ANALYZE RESULTS
# ============================================================

def analyze_handwashing(predictions, confidences, seq_start_frames):
    """
    Analyze if handwashing was done correctly according to WHO guidelines.
    """
    print("\n" + "="*70)
    print("📊 HANDWASHING ANALYSIS")
    print("="*70)
    
    # Count occurrences of each step
    step_counts = Counter(predictions)
    
    # Identify which WHO steps were performed
    performed_steps = set()
    for pred in predictions:
        if pred > 0:  # Exclude background
            performed_steps.add(pred)
    
    # Check if all required steps were performed
    missing_steps = set(REQUIRED_STEPS) - performed_steps
    
    # Calculate average confidence
    avg_confidence = confidences.mean() * 100
    
    # Determine if handwashing was correct
    is_correct = len(missing_steps) == 0
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total sequences analyzed: {len(predictions)}")
    print(f"   Average confidence: {avg_confidence:.1f}%")
    print(f"   Unique steps detected: {len(performed_steps)}")
    
    print(f"\n🔍 Step-by-Step Breakdown:")
    print(f"{'WHO Step':<45} {'Detected':<10} {'Sequences':<10} {'Duration'}")
    print("-" * 70)
    
    for step_num in range(9):
        count = step_counts.get(step_num, 0)
        duration = count * SEQ_LEN / 30  # Convert to seconds (30 FPS)
        detected = "✓" if step_num in performed_steps or step_num == 0 else "✗"
        
        if count > 0:
            print(f"{WHO_STEPS[step_num]:<45} {detected:<10} {count:<10} {duration:.1f}s")
    
    print("\n" + "="*70)
    print("🎯 WHO COMPLIANCE CHECK")
    print("="*70)
    
    print(f"\nRequired Steps: {len(REQUIRED_STEPS)}")
    print(f"Performed Steps: {len(performed_steps)}")
    
    if is_correct:
        print("\n✅ CORRECT HANDWASHING!")
        print("   All WHO required steps (1-7) were performed.")
    else:
        print("\n❌ INCORRECT HANDWASHING!")
        print(f"   Missing {len(missing_steps)} required step(s):")
        for step in sorted(missing_steps):
            print(f"   - {WHO_STEPS[step]}")
    
    # Check for proper sequence
    print(f"\n📋 Step Sequence:")
    sequence_display = []
    prev_step = None
    
    for pred in predictions:
        if pred != prev_step and pred != 0:  # New step (skip background)
            sequence_display.append(WHO_STEPS[pred].split(":")[0])
            prev_step = pred
    
    if sequence_display:
        print("   → " + " → ".join(sequence_display))
    else:
        print("   ⚠️  No clear washing steps detected (only background)")
    
    print("\n" + "="*70)
    
    return is_correct, missing_steps, step_counts

# ============================================================
# VISUALIZATION
# ============================================================

def visualize_timeline(predictions, seq_start_frames, save_path="test_results_timeline.png"):
    """
    Create a timeline visualization of predicted steps.
    """
    print(f"\n📊 Generating timeline visualization...")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Timeline plot
    times = np.array(seq_start_frames) / 30.0  # Convert frames to seconds
    colors = plt.cm.tab10(predictions / 8.0)
    
    ax1.scatter(times, predictions, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('WHO Step', fontsize=12)
    ax1.set_title('Handwashing Timeline - Detected WHO Steps', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(9))
    ax1.set_yticklabels([f"Step {i}" if i > 0 else "Background" for i in range(9)])
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 8.5)
    
    # Bar chart of step durations
    step_counts = Counter(predictions)
    steps = sorted([s for s in step_counts.keys() if s > 0])
    counts = [step_counts[s] for s in steps]
    durations = [c * SEQ_LEN / 30 for c in counts]  # Convert to seconds
    
    colors_bar = [plt.cm.tab10(s / 8.0) for s in steps]
    bars = ax2.bar(range(len(steps)), durations, color=colors_bar, edgecolor='black', alpha=0.7)
    
    ax2.set_xlabel('WHO Step', fontsize=12)
    ax2.set_ylabel('Duration (seconds)', fontsize=12)
    ax2.set_title('Duration of Each WHO Step', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels([f"Step {s}" for s in steps])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, dur) in enumerate(zip(bars, durations)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{dur:.1f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Timeline saved to: {save_path}")
    
    return save_path

# ============================================================
# GENERATE DETAILED REPORT
# ============================================================

def generate_report(video_path, predictions, confidences, seq_start_frames, 
                   is_correct, missing_steps, step_counts, save_path="test_results_report.txt"):
    """
    Generate a detailed text report.
    """
    print(f"\n📝 Generating detailed report...")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("WHO HANDWASHING COMPLIANCE TEST REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Video Analyzed: {video_path}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System: Ensemble LSTM (5 models, 90.83% accuracy)\n\n")
        
        f.write("-"*70 + "\n")
        f.write("OVERALL RESULT\n")
        f.write("-"*70 + "\n\n")
        
        if is_correct:
            f.write("✅ PASSED - Handwashing performed correctly\n\n")
            f.write("All 7 WHO required steps were detected:\n")
            for step in REQUIRED_STEPS:
                f.write(f"  ✓ {WHO_STEPS[step]}\n")
        else:
            f.write("❌ FAILED - Handwashing incomplete\n\n")
            f.write(f"Missing {len(missing_steps)} required step(s):\n")
            for step in sorted(missing_steps):
                f.write(f"  ✗ {WHO_STEPS[step]}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("DETAILED ANALYSIS\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"Total Sequences: {len(predictions)}\n")
        f.write(f"Average Confidence: {confidences.mean()*100:.1f}%\n")
        f.write(f"Video Duration: {seq_start_frames[-1]/30:.1f} seconds\n\n")
        
        f.write("Step-by-Step Breakdown:\n")
        f.write(f"{'Step':<45} {'Sequences':<12} {'Duration':<12} {'Avg Confidence'}\n")
        f.write("-"*70 + "\n")
        
        for step_num in range(9):
            count = step_counts.get(step_num, 0)
            if count > 0:
                duration = count * SEQ_LEN / 30
                # Get confidence for this step
                step_confidences = confidences[predictions == step_num]
                avg_conf = step_confidences.mean() * 100 if len(step_confidences) > 0 else 0
                
                f.write(f"{WHO_STEPS[step_num]:<45} {count:<12} {duration:<12.1f} {avg_conf:.1f}%\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-"*70 + "\n\n")
        
        if is_correct:
            f.write("• Excellent handwashing technique!\n")
            f.write("• All WHO recommended steps completed.\n")
            f.write("• Continue maintaining this standard.\n")
        else:
            f.write("• Please ensure all 7 WHO steps are performed.\n")
            if missing_steps:
                f.write("• Focus on practicing these missing steps:\n")
                for step in sorted(missing_steps):
                    f.write(f"  - {WHO_STEPS[step]}\n")
            f.write("• Refer to WHO handwashing guidelines for proper technique.\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✅ Report saved to: {save_path}")
    return save_path

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    import pandas as pd  # For timestamp in report
    
    print("="*70)
    print("🧼 WHO HANDWASHING COMPLIANCE TESTER")
    print("="*70)
    print("This tool analyzes handwashing videos to determine if WHO")
    print("7-step handwashing technique was performed correctly.")
    print("="*70 + "\n")
    
    # Get video path from user
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("Enter path to handwashing video: ").strip().strip('"')
    
    # Validate video exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Load ensemble models
    models = load_ensemble()
    
    # Process video
    predictions, confidences, seq_start_frames = process_video(video_path, models)
    
    if predictions is None:
        return
    
    # Analyze results
    is_correct, missing_steps, step_counts = analyze_handwashing(
        predictions, confidences, seq_start_frames
    )
    
    # Generate visualizations
    timeline_path = visualize_timeline(predictions, seq_start_frames)
    
    # Generate report
    report_path = generate_report(
        video_path, predictions, confidences, seq_start_frames,
        is_correct, missing_steps, step_counts
    )
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved:")
    print(f"  📊 Timeline: {timeline_path}")
    print(f"  📝 Report:   {report_path}")
    
    if is_correct:
        print(f"\n🎉 Result: CORRECT HANDWASHING ✅")
    else:
        print(f"\n⚠️  Result: INCOMPLETE HANDWASHING ❌")
        print(f"   Missing {len(missing_steps)} step(s)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()