"""
🔥 CRITICAL: Run this script FIRST to regenerate sequences with fixed labeling!

This script:
1. Loads annotations with majority voting
2. Interpolates missing frames
3. Extracts NORMALIZED features
4. Labels sequences using MAJORITY VOTE across frames
5. Saves cleaned data for training

Usage:
    python preprocess_data.py
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Get project root (where this script is located)
PROJECT_ROOT = Path(__file__).resolve().parent

# Add src directory to Python path
sys.path.insert(0, str(PROJECT_ROOT))

# Import from src package
from src.utils.annotation_loader import load_annotations, get_label_for_sequence, interpolate_missing_frames
from src.feature_extractor import extract_features, SEQ_LEN

# -------- CONFIG --------
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("🔧 Handwash Data Preprocessing Pipeline")
print("=" * 60)
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Video dir: {VIDEO_DIR}")
print(f"📁 Annotation dir: {ANNOTATION_DIR}")
print(f"📁 Output dir: {OUTPUT_DIR}")

# Verify directories exist
if not VIDEO_DIR.exists():
    print(f"❌ ERROR: Video directory not found: {VIDEO_DIR}")
    sys.exit(1)
    
if not ANNOTATION_DIR.exists():
    print(f"❌ ERROR: Annotation directory not found: {ANNOTATION_DIR}")
    sys.exit(1)

# -------- LOAD ANNOTATIONS --------
print("\n📋 Step 1: Loading annotations from all annotators...")
try:
    annotations = load_annotations(ANNOTATION_DIR)
    print(f"✅ Loaded annotations for {len(annotations)} videos")
except Exception as e:
    print(f"❌ ERROR loading annotations: {e}")
    sys.exit(1)

# -------- INTERPOLATE MISSING FRAMES --------
print("\n🔗 Step 2: Interpolating missing frames...")
for video_name in annotations:
    original_count = len(annotations[video_name])
    annotations[video_name] = interpolate_missing_frames(annotations[video_name])
    new_count = len(annotations[video_name])
    if new_count > original_count:
        print(f"  {video_name}: {original_count} → {new_count} frames")

# -------- EXTRACT FEATURES --------
print("\n🎥 Step 3: Extracting features from videos...")

all_sequences = []
all_labels = []
all_video_ids = []

video_files = sorted(VIDEO_DIR.glob("*.mp4"))

if len(video_files) == 0:
    print("❌ No videos found! Check your VIDEO_DIR path.")
    print(f"   Looking in: {VIDEO_DIR}")
    sys.exit(1)

print(f"Found {len(video_files)} videos")

for video_path in tqdm(video_files, desc="Processing videos"):
    video_name = video_path.stem
    
    # Extract features
    try:
        result = extract_features(video_path, return_frames=True)
    except Exception as e:
        print(f"  ❌ Error processing {video_name}: {e}")
        continue
    
    if result is None:
        print(f"  ⚠️  Skipped {video_name}: insufficient frames")
        continue
    
    sequences, seq_start_frames = result
    
    # Get labels for this video
    if video_name not in annotations:
        print(f"  ⚠️  Skipped {video_name}: no annotations")
        continue
    
    frame_labels = annotations[video_name]
    
    # 🔥 FIXED: Label each sequence using majority vote
    sequences_added = 0
    for i, start_frame in enumerate(seq_start_frames):
        label = get_label_for_sequence(start_frame, SEQ_LEN, frame_labels)
        
        if label == -1:  # Skip unlabeled sequences
            continue
        
        all_sequences.append(sequences[i])
        all_labels.append(label)
        all_video_ids.append(video_name)
        sequences_added += 1
    
    if sequences_added > 0:
        print(f"  ✅ {video_name}: {sequences_added} sequences extracted")

# -------- SAVE PROCESSED DATA --------
print("\n💾 Step 4: Saving processed data...")

if len(all_sequences) == 0:
    print("❌ ERROR: No sequences extracted! Check your videos and annotations.")
    sys.exit(1)

all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)
all_video_ids = np.array(all_video_ids)

np.save(OUTPUT_DIR / "sequences_X.npy", all_sequences)
np.save(OUTPUT_DIR / "sequences_y.npy", all_labels)
np.save(OUTPUT_DIR / "video_ids.npy", all_video_ids)

print("✅ Saved to outputs/")
print(f"  sequences_X.npy: {all_sequences.shape}")
print(f"  sequences_y.npy: {all_labels.shape}")
print(f"  video_ids.npy: {len(np.unique(all_video_ids))} unique videos")

# -------- STATISTICS --------
print("\n" + "=" * 60)
print("📊 Dataset Statistics")
print("=" * 60)

unique_labels, counts = np.unique(all_labels, return_counts=True)
print(f"\nTotal sequences: {len(all_sequences)}")
print(f"Total videos processed: {len(np.unique(all_video_ids))}")
print(f"\nClass Distribution:")
for label, count in zip(unique_labels, counts):
    percentage = 100 * count / len(all_labels)
    print(f"  Class {label}: {count:5d} sequences ({percentage:5.2f}%)")

# Check for severe imbalance
max_count = counts.max()
min_count = counts.min()
imbalance_ratio = max_count / min_count

print(f"\n⚖️  Imbalance Ratio: {imbalance_ratio:.1f}x")
if imbalance_ratio > 20:
    print("  ⚠️  SEVERE IMBALANCE DETECTED!")
    print("  Consider data augmentation for rare classes")
elif imbalance_ratio > 10:
    print("  ⚠️  Moderate imbalance - class weights will help")
else:
    print("  ✅ Manageable imbalance")

print("\n" + "=" * 60)
print("✅ Preprocessing Complete!")
print("=" * 60)
print("\n🚀 Next step: Run 'python src/train_lstm.py'")