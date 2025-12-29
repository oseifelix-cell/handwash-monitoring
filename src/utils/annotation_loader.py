import csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

FPS = 30  # adjust if your videos differ


def load_annotations(annotation_root: Path):
    """
    Load annotations from ALL annotators and apply majority voting.

    Returns:
        annotations[video_name][frame_idx] = label (0–7)
    """
    annotator_dirs = [d for d in annotation_root.iterdir() if d.is_dir()]

    # video → frame → list of labels from annotators
    votes = defaultdict(lambda: defaultdict(list))

    for annot_dir in annotator_dirs:
        for csv_file in annot_dir.glob("*.csv"):
            video_name = csv_file.stem

            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        time_ms = float(row["frame_time"])
                        is_washing = int(row["is_washing"])
                        movement = int(row["movement_code"])
                    except ValueError:
                        continue

                    frame_idx = int((time_ms / 1000) * FPS)

                    # WHO labeling logic
                    if is_washing == 0:
                        label = 0  # background / no washing
                    else:
                        label = movement + 1  # WHO steps 1–7

                    votes[video_name][frame_idx].append(label)

    # Majority vote
    final_annotations = {}

    for video, frame_votes in votes.items():
        frame_labels = {}
        for frame, labels in frame_votes.items():
            label = Counter(labels).most_common(1)[0][0]
            frame_labels[frame] = label
        final_annotations[video] = frame_labels

    return final_annotations


def get_label_for_sequence(start_frame, seq_len, frame_label_dict):
    """
    🔥 FIXED: Assign label using MAJORITY VOTE across the entire sequence.
    
    Args:
        start_frame: First frame of the sequence
        seq_len: Length of sequence (e.g., 30)
        frame_label_dict: Dict mapping frame_idx -> label
    
    Returns:
        Most common label in the sequence, or -1 if no labels found
    """
    labels_in_sequence = []
    
    # Collect all labels within the sequence window
    for frame_idx in range(start_frame, start_frame + seq_len):
        if frame_idx in frame_label_dict:
            labels_in_sequence.append(frame_label_dict[frame_idx])
    
    # If we have labels, return the most common one
    if labels_in_sequence:
        return Counter(labels_in_sequence).most_common(1)[0][0]
    
    # If no labels in this sequence, search nearby frames (tolerance ±5)
    for offset in range(1, 6):
        check_frames = [start_frame - offset, start_frame + seq_len + offset]
        for frame in check_frames:
            if frame in frame_label_dict:
                return frame_label_dict[frame]
    
    return -1  # unlabeled


def interpolate_missing_frames(frame_label_dict, max_gap=10):
    """
    🔥 NEW: Fill small gaps in annotations using forward-fill strategy.
    
    This handles cases where annotators skipped a few frames between actions.
    """
    if not frame_label_dict:
        return frame_label_dict
    
    frame_indices = sorted(frame_label_dict.keys())
    interpolated = frame_label_dict.copy()
    
    for i in range(len(frame_indices) - 1):
        current_frame = frame_indices[i]
        next_frame = frame_indices[i + 1]
        gap = next_frame - current_frame
        
        # Only interpolate small gaps
        if 1 < gap <= max_gap:
            current_label = frame_label_dict[current_frame]
            # Forward fill with current label
            for fill_frame in range(current_frame + 1, next_frame):
                interpolated[fill_frame] = current_label
    
    return interpolated