import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

SEQ_LEN = 30          # frames per sequence
STRIDE = 10           # sliding window step
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # x,y,z


def normalize_landmarks(landmarks):
    """
    🔥 NEW: Normalize landmarks to be position/scale invariant.
    
    Strategy:
    1. Center at wrist (landmark 0)
    2. Scale by hand size (max distance from wrist)
    """
    landmarks = np.array(landmarks).reshape(NUM_LANDMARKS, 3)
    
    # Center at wrist
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist
    
    # Scale by hand size
    distances = np.linalg.norm(landmarks, axis=1)
    max_dist = np.max(distances)
    
    if max_dist > 0:
        landmarks = landmarks / max_dist
    
    return landmarks.flatten()


def extract_features(video_path, return_frames=False):
    """
    Extract normalized hand landmarks from video.
    
    Returns:
        sequences: (N, SEQ_LEN, FEATURE_DIM) - temporal sequences
        seq_start_frames: (N,) - frame indices for labeling
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None

    all_landmarks = []
    frame_indices = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                features = []
                for p in lm:
                    features.extend([p.x, p.y, p.z])
                
                # 🔥 APPLY NORMALIZATION
                normalized = normalize_landmarks(features)
                all_landmarks.append(normalized)
                frame_indices.append(frame_idx)

            frame_idx += 1

    cap.release()

    # Not enough frames to form one sequence
    if len(all_landmarks) < SEQ_LEN:
        return None

    all_landmarks = np.array(all_landmarks)

    sequences = []
    seq_start_frames = []

    for i in range(0, len(all_landmarks) - SEQ_LEN + 1, STRIDE):
        seq = all_landmarks[i:i + SEQ_LEN]

        # Safety check
        if seq.shape != (SEQ_LEN, FEATURE_DIM):
            continue

        sequences.append(seq)
        seq_start_frames.append(frame_indices[i])

    # No valid sequences extracted
    if len(sequences) == 0:
        return None

    sequences = np.array(sequences)
    seq_start_frames = np.array(seq_start_frames)

    if return_frames:
        return sequences, seq_start_frames
    else:
        return sequences