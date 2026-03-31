"""
EchoFind — Out-of-Distribution Detection via Mahalanobis Distance

Fits a class-conditional Gaussian model over encoder embeddings and
flags inputs whose Mahalanobis distance to every class centroid exceeds
a learned threshold as *Unknown / Anomaly*.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from scipy.spatial.distance import mahalanobis

from .model import AudioEncoder


# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------

def generate_white_noise(
    duration: float = 30.0, sr: int = 22_050, amplitude: float = 0.5
) -> np.ndarray:
    """Return white Gaussian noise as a 1-D float32 array."""
    n = int(sr * duration)
    return (np.random.randn(n) * amplitude).astype(np.float32)


def generate_sine(
    freq: float = 440.0,
    duration: float = 30.0,
    sr: int = 22_050,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Return a pure sine tone as a 1-D float32 array."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def generate_chirp(
    f_start: float = 20.0,
    f_end: float = 8000.0,
    duration: float = 30.0,
    sr: int = 22_050,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Return a linear chirp sweep as a 1-D float32 array."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    phase = 2 * np.pi * (f_start + (f_end - f_start) / (2 * duration) * t) * t
    return (np.sin(phase) * amplitude).astype(np.float32)


# ---------------------------------------------------------------------------
# Signal → embedding helper
# ---------------------------------------------------------------------------

def signal_to_embedding(
    signal: np.ndarray,
    encoder: nn.Module,
    device: torch.device,
    sr: int = 22_050,
    target_frames: int = 1292,
    center_start: int = 538,
    center_end: int = 754,
) -> np.ndarray:
    """Convert a raw 1-D waveform to an L2-normalised 512-d embedding.

    Internally computes a log-mel spectrogram with the same parameters
    used during pre-training, takes a center crop, and runs the frozen
    encoder.

    Args:
        signal:        1-D float32 waveform.
        encoder:       Frozen ``AudioEncoder``.
        device:        CUDA / CPU device.
        sr:            Sample rate (must match training: 22 050).
        target_frames: Number of spectrogram frames to pad/crop to.
        center_start:  Start frame of the center crop.
        center_end:    End frame of the center crop.

    Returns:
        Unit-normalised numpy array of shape ``(512,)``.
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=128, hop_length=512, n_fft=2048
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    waveform = torch.from_numpy(signal).float().unsqueeze(0)  # (1, N)
    log_mel = amp_to_db(mel_transform(waveform))              # (1, 128, T)

    T = log_mel.shape[2]
    if T < target_frames:
        pad = target_frames - T
        log_mel = torch.nn.functional.pad(
            log_mel, (0, pad), mode="constant", value=log_mel.min().item()
        )
    else:
        log_mel = log_mel[:, :, :target_frames]

    crop = log_mel[:, :, center_start:center_end]             # (1, 128, 216)
    tensor = crop.unsqueeze(0).to(device)                     # (1, 1, 128, 216)

    with torch.no_grad():
        h = encoder(tensor)
    h = h.cpu().numpy().flatten()
    h /= np.linalg.norm(h) + 1e-8
    return h


# ---------------------------------------------------------------------------
# Mahalanobis OOD detector
# ---------------------------------------------------------------------------

class MahalanobisOOD:
    """Class-conditional Mahalanobis distance OOD detector.

    Fits per-class centroids and a shared (regularised) covariance matrix
    from in-distribution training embeddings.  At inference time, an input
    whose minimum Mahalanobis distance to all centroids exceeds the 95th
    percentile of the training distribution is flagged as OOD.

    Args:
        percentile: Training-score percentile used as the OOD threshold
                    (default: 95).
        reg:        Covariance regularisation coefficient (default: 1e-5).
    """

    def __init__(self, percentile: float = 95.0, reg: float = 1e-5) -> None:
        self.percentile = percentile
        self.reg = reg
        self.class_means: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.genre_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit the detector on in-distribution training embeddings.

        Args:
            embeddings: Array of shape ``(N, D)`` — L2-normalised.
            labels:     String array of shape ``(N,)`` — genre labels.
        """
        self.genre_names = sorted(set(labels))
        dim = embeddings.shape[1]

        # Per-class centroids
        self.class_means = np.zeros(
            (len(self.genre_names), dim), dtype=np.float64
        )
        for i, genre in enumerate(self.genre_names):
            self.class_means[i] = embeddings[labels == genre].mean(axis=0)

        # Shared covariance + regularisation
        cov = np.cov(embeddings.T)
        cov += self.reg * np.eye(dim)
        self.cov_inv = np.linalg.inv(cov)

        # Threshold from training scores
        train_scores = np.array(
            [self._min_mahal(e) for e in embeddings]
        )
        self.threshold = float(np.percentile(train_scores, self.percentile))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _min_mahal(self, embedding: np.ndarray) -> float:
        """Minimum Mahalanobis distance to any class centroid."""
        return min(
            mahalanobis(embedding.astype(np.float64), cm, self.cov_inv)
            for cm in self.class_means
        )

    def score(self, embedding: np.ndarray) -> float:
        """Return the Mahalanobis OOD score (lower = more in-distribution).

        Args:
            embedding: 1-D array of shape ``(D,)``.

        Returns:
            Minimum Mahalanobis distance to any class centroid.
        """
        return self._min_mahal(embedding)

    def predict(
        self, embedding: np.ndarray
    ) -> Tuple[str, float]:
        """Classify an embedding or flag it as OOD.

        Args:
            embedding: 1-D array of shape ``(D,)``.

        Returns:
            Tuple of ``(predicted_label, mahalanobis_score)``.
            If the score exceeds the threshold the label is
            ``"Unknown/Anomaly"``.
        """
        dists = [
            mahalanobis(embedding.astype(np.float64), cm, self.cov_inv)
            for cm in self.class_means
        ]
        min_dist = min(dists)
        if min_dist >= self.threshold:
            return "Unknown/Anomaly", min_dist
        nearest = int(np.argmin(dists))
        return self.genre_names[nearest], min_dist
