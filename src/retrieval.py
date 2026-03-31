"""
EchoFind — FAISS Retrieval Engine

Builds and queries a cosine-similarity index over L2-normalised
encoder embeddings for audio fingerprint retrieval.
"""

import os
from typing import List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import fftconvolve

from .model import AudioEncoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CENTER_START: int = 538   # center-crop start frame for a 1292-frame spectrogram
CENTER_END: int = 754     # center-crop end   frame  (538 + 216 = 754)
EMBED_DIM: int = 512


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def extract_embedding(
    spec: np.ndarray,
    encoder: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Extract an L2-normalised embedding from a full spectrogram.

    Args:
        spec:    Numpy array of shape ``(1, 128, 1292)``.
        encoder: Frozen ``AudioEncoder`` (or DataParallel-wrapped).
        device:  Target CUDA / CPU device.

    Returns:
        1-D numpy array of shape ``(512,)``, unit-normalised.
    """
    crop = spec[:, :, CENTER_START:CENTER_END].copy()
    tensor = torch.from_numpy(crop).float().unsqueeze(0).to(device)
    with torch.no_grad():
        h = encoder(tensor)
    h = h.cpu().numpy().flatten()
    h /= np.linalg.norm(h) + 1e-8
    return h


def _add_channel_noise(
    crop: np.ndarray, snr_db: float = 5.0
) -> np.ndarray:
    """Add Gaussian noise at a target SNR and mild exponential-decay reverb.

    Args:
        crop:   Numpy array of shape ``(1, 128, T)``.
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        Degraded spectrogram crop (same shape).
    """
    signal_std = np.std(crop)
    noise_std = signal_std / (10 ** (snr_db / 20))
    crop = crop + np.random.randn(*crop.shape).astype(np.float32) * noise_std

    # Mild reverb via exponential-decay impulse response
    ir = np.exp(-np.linspace(0, 5, 2000)).astype(np.float32)
    ir /= ir.sum()
    reverbed = np.zeros_like(crop)
    for c in range(crop.shape[0]):
        for m in range(crop.shape[1]):
            conv = fftconvolve(crop[c, m, :], ir, mode="full")
            reverbed[c, m, :] = conv[: crop.shape[2]]
    return reverbed


# ---------------------------------------------------------------------------
# FAISS Retriever
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """Cosine-similarity retrieval engine backed by FAISS IndexFlatIP.

    Typical workflow::

        retriever = FAISSRetriever(encoder, device)
        retriever.build_index(track_ids, specs_dir)
        pred_id, score = retriever.query(spec_path)
        acc = retriever.evaluate(track_ids, specs_dir, add_noise=True)

    Args:
        encoder: Frozen ``AudioEncoder`` (may be DataParallel-wrapped).
        device:  CUDA / CPU device for inference.
    """

    def __init__(self, encoder: nn.Module, device: torch.device) -> None:
        self.encoder = encoder
        self.device = device
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_list: List[int] = []

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(
        self, track_ids: List[int], specs_dir: str
    ) -> None:
        """Build a FAISS inner-product index from spectrogram files.

        Args:
            track_ids: List of track IDs to index.
            specs_dir: Directory containing ``{track_id}.npy`` files.
        """
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.id_list = []

        for tid in track_ids:
            spec = np.load(
                os.path.join(specs_dir, f"{tid}.npy")
            ).astype(np.float32)
            emb = extract_embedding(spec, self.encoder, self.device)
            self.index.add(emb.reshape(1, -1).astype(np.float32))
            self.id_list.append(tid)

    def save(self, index_path: str, id_map_path: str) -> None:
        """Persist index and ID map to disk.

        Args:
            index_path:  Output path for the FAISS binary index.
            id_map_path: Output path for the numpy ID map.
        """
        faiss.write_index(self.index, index_path)
        np.save(id_map_path, np.array(self.id_list))

    def load(self, index_path: str, id_map_path: str) -> None:
        """Load a previously saved index and ID map.

        Args:
            index_path:  Path to the FAISS binary index.
            id_map_path: Path to the numpy ID map.
        """
        self.index = faiss.read_index(index_path)
        self.id_list = np.load(id_map_path).tolist()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        spec_path: str,
        k: int = 1,
        add_noise: bool = False,
    ) -> Tuple[int, float]:
        """Query the index with a spectrogram file.

        Args:
            spec_path: Path to a ``.npy`` spectrogram of shape ``(1, 128, 1292)``.
            k:         Number of nearest neighbours (only top-1 returned).
            add_noise: If True, degrade the query with noise + reverb.

        Returns:
            Tuple of ``(predicted_track_id, cosine_similarity)``.
        """
        spec = np.load(spec_path).astype(np.float32)
        crop = spec[:, :, CENTER_START:CENTER_END].copy()

        if add_noise:
            crop = _add_channel_noise(crop)

        tensor = torch.from_numpy(crop).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            h = self.encoder(tensor)
        h = h.cpu().numpy().flatten()
        h /= np.linalg.norm(h) + 1e-8

        D, I = self.index.search(
            h.reshape(1, -1).astype(np.float32), k
        )
        return self.id_list[I[0, 0]], float(D[0, 0])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        track_ids: List[int],
        specs_dir: str,
        add_noise: bool = False,
    ) -> float:
        """Compute top-1 retrieval accuracy over a set of tracks.

        Args:
            track_ids: Track IDs to evaluate (must already be in the index).
            specs_dir: Directory containing spectrogram files.
            add_noise: Whether to degrade queries with noise + reverb.

        Returns:
            Retrieval accuracy as a fraction in [0, 1].
        """
        correct = 0
        for tid in track_ids:
            spec_path = os.path.join(specs_dir, f"{tid}.npy")
            pred_id, _ = self.query(spec_path, add_noise=add_noise)
            if pred_id == tid:
                correct += 1
        return correct / len(track_ids)
