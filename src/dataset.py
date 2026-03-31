"""
EchoFind — FMA Dataset and Augmentation Pipeline

Provides a PyTorch Dataset for SimCLR contrastive learning on
pre-computed log-mel spectrograms from the FMA-Small corpus.
"""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_FRAMES: int = 1292       # 30 s @ sr=22050, hop=512
CROP_FRAMES: int = 216         # ~5 s crop
FREQ_MASK_MAX: int = 30        # SpecAugment frequency mask width
TIME_MASK_MAX: int = 50        # SpecAugment time mask width
NOISE_STD: float = 0.01        # Gaussian noise standard deviation


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_track_ids(
    metadata_dir: str,
    split: str = "training",
    subset: str = "small",
) -> List[int]:
    """Load track IDs for a given FMA split.

    Args:
        metadata_dir: Path to the FMA metadata directory containing tracks.csv.
        split:        One of 'training', 'validation', 'test'.
        subset:       FMA subset name (default: 'small').

    Returns:
        Sorted list of integer track IDs.
    """
    tracks = pd.read_csv(
        os.path.join(metadata_dir, "tracks.csv"), index_col=0, header=[0, 1]
    )
    mask = (tracks[("set", "subset")] == subset) & (
        tracks[("set", "split")] == split
    )
    return sorted(tracks[mask].index.tolist())


def load_genre_labels(
    metadata_dir: str, track_ids: List[int]
) -> np.ndarray:
    """Return genre labels for the given track IDs.

    Args:
        metadata_dir: Path to FMA metadata directory.
        track_ids:    List of integer track IDs.

    Returns:
        Numpy string array of genre labels aligned with *track_ids*.
    """
    tracks = pd.read_csv(
        os.path.join(metadata_dir, "tracks.csv"), index_col=0, header=[0, 1]
    )
    return tracks.loc[track_ids, ("track", "genre_top")].values.astype(str)


def load_corrupt_ids(metadata_dir: str) -> set:
    """Load the set of corrupt / missing track IDs from not_found.pickle.

    Args:
        metadata_dir: Path to FMA metadata directory.

    Returns:
        Set of integer track IDs known to be corrupt or missing.
    """
    path = os.path.join(metadata_dir, "not_found.pickle")
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as fh:
        not_found = pickle.load(fh)
    return set(int(x) for x in not_found.get("audio", []))


def filter_existing(
    track_ids: List[int], specs_dir: str
) -> List[int]:
    """Keep only track IDs whose spectrogram .npy file exists on disk.

    Args:
        track_ids: Candidate track IDs.
        specs_dir: Directory containing ``{track_id}.npy`` files.

    Returns:
        Filtered list of track IDs.
    """
    return [
        tid
        for tid in track_ids
        if os.path.exists(os.path.join(specs_dir, f"{tid}.npy"))
    ]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def spec_augment(tensor: torch.Tensor) -> torch.Tensor:
    """Apply SpecAugment + additive Gaussian noise in-place.

    Augmentations applied sequentially:
        1. Frequency masking — zero out up to 30 consecutive mel bands.
        2. Time masking     — zero out up to 50 consecutive time frames.
        3. Gaussian noise   — additive N(0, 0.01).

    Args:
        tensor: Spectrogram crop of shape (1, 128, 216).

    Returns:
        Augmented tensor (same shape, float32).
    """
    _, n_mels, n_frames = tensor.shape

    # Frequency mask
    f_start = torch.randint(0, n_mels - FREQ_MASK_MAX, (1,)).item()
    f_width = torch.randint(0, FREQ_MASK_MAX + 1, (1,)).item()
    tensor[:, f_start : f_start + f_width, :] = 0.0

    # Time mask
    t_start = torch.randint(0, n_frames - TIME_MASK_MAX, (1,)).item()
    t_width = torch.randint(0, TIME_MASK_MAX + 1, (1,)).item()
    tensor[:, :, t_start : t_start + t_width] = 0.0

    # Gaussian noise
    tensor = tensor + torch.randn_like(tensor) * NOISE_STD

    return tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FMAContrastiveDataset(Dataset):
    """SimCLR contrastive dataset over pre-computed log-mel spectrograms.

    Each ``__getitem__`` call returns two independently cropped and augmented
    views of the same track, suitable for the NT-Xent objective.

    Args:
        track_ids: List of valid integer track IDs.
        specs_dir: Directory containing ``{track_id}.npy`` spectrograms
                   of shape ``(1, 128, 1292)``.
        augment:   Whether to apply SpecAugment + noise (disable for val).
    """

    def __init__(
        self,
        track_ids: List[int],
        specs_dir: str,
        augment: bool = True,
    ) -> None:
        self.track_ids = track_ids
        self.specs_dir = specs_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.track_ids)

    def _random_crop(self, spec: np.ndarray) -> np.ndarray:
        """Sample a random 5-second crop from a full spectrogram."""
        max_start = TOTAL_FRAMES - CROP_FRAMES
        start = np.random.randint(0, max_start + 1)
        return spec[:, :, start : start + CROP_FRAMES].copy()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented views of the same track.

        Returns:
            Tuple of (view1, view2), each of shape (1, 128, 216).
        """
        tid = self.track_ids[idx]
        spec = np.load(
            os.path.join(self.specs_dir, f"{tid}.npy")
        ).astype(np.float32)

        crop1 = torch.from_numpy(self._random_crop(spec))
        crop2 = torch.from_numpy(self._random_crop(spec))

        if self.augment:
            crop1 = spec_augment(crop1)
            crop2 = spec_augment(crop2)

        return crop1, crop2
