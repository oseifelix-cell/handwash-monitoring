import numpy as np
import torch


def temporal_warp(sequence, magnitude=0.2):
    """Time warp: stretch/compress temporal dimension"""
    seq_len = len(sequence)
    indices = np.arange(seq_len)
    
    # Random warping
    warp = np.random.normal(0, magnitude, seq_len).cumsum()
    warped_indices = indices + warp
    warped_indices = np.clip(warped_indices, 0, seq_len - 1).astype(int)
    
    return sequence[warped_indices]


def add_noise(sequence, noise_level=0.02):
    """Add Gaussian noise to landmarks"""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise


def random_scale(sequence, scale_range=(0.9, 1.1)):
    """Random scaling of hand size"""
    scale = np.random.uniform(*scale_range)
    return sequence * scale


def random_rotation(sequence, max_angle=15):
    """Rotate hand in 2D plane"""
    angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # Reshape to (seq_len, 21, 3)
    seq = sequence.reshape(-1, 21, 3)
    
    # Rotate x,y coordinates
    x, y, z = seq[:, :, 0], seq[:, :, 1], seq[:, :, 2]
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    
    seq[:, :, 0] = x_rot
    seq[:, :, 1] = y_rot
    
    return seq.reshape(-1, 63)


def augment_sequence(sequence, label, augmentation_prob=0.8):
    """Apply random augmentations"""
    if np.random.random() > augmentation_prob:
        return sequence
    
    # Don't augment background class too much
    if label == 0:
        augmentation_prob = 0.3
    
    aug_seq = sequence.copy()
    
    # Apply 2-3 random augmentations
    if np.random.random() > 0.5:
        aug_seq = temporal_warp(aug_seq)
    if np.random.random() > 0.5:
        aug_seq = add_noise(aug_seq)
    if np.random.random() > 0.5:
        aug_seq = random_scale(aug_seq)
    if np.random.random() > 0.5:
        aug_seq = random_rotation(aug_seq)
    
    return aug_seq


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset with online augmentation"""
    def __init__(self, X, y, augment=True):
        self.X = X
        self.y = y
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            x = augment_sequence(x, y)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)