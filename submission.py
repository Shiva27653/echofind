"""
EchoFind — Submission Interface

Provides a self-contained inference API for the IEEE competition.
All necessary components (encoder architecture, embedding extraction,
FAISS retrieval) are defined locally so this file can be used
independently of the ``src/`` package.

Usage::

    from submission import predict_track, get_embedding

    embedding = get_embedding("/path/to/query.npy")
    track_id, score = predict_track("/path/to/query.npy")
"""

import os
from typing import Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# Encoder (self-contained — mirrors src/model.py)
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """ResNet-18 encoder for single-channel log-mel spectrograms.

    Architecture:
        - ``conv1``: 1-channel input, 64 filters, 7×7, stride 2, padding 3
        - ``fc``: replaced with ``nn.Identity()`` → 512-d output
    """

    def __init__(self) -> None:
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, 1, 128, T)``.

        Returns:
            Feature tensor of shape ``(B, 512)``.
        """
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Global state (lazy-loaded on first call)
# ---------------------------------------------------------------------------

_encoder: nn.Module = None
_index: faiss.IndexFlatIP = None
_id_list: np.ndarray = None
_device: torch.device = None

# Default paths — override via environment variables if needed
_WEIGHTS_PATH = os.environ.get(
    "ECHOFIND_WEIGHTS", "weights/encoder.pth"
)
_INDEX_PATH = os.environ.get(
    "ECHOFIND_INDEX", "faiss_index.bin"
)
_ID_MAP_PATH = os.environ.get(
    "ECHOFIND_ID_MAP", "id_map.npy"
)
_DEVICE_IDS = [0, 1]

# Crop constants (center 5 s of a 30 s spectrogram)
_CENTER_START = 538
_CENTER_END = 754


def _ensure_loaded() -> None:
    """Lazy-load the encoder, FAISS index, and ID map on first use."""
    global _encoder, _index, _id_list, _device

    if _encoder is not None:
        return

    torch.cuda.set_device(_DEVICE_IDS[0])
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encoder
    encoder = AudioEncoder()
    encoder.load_state_dict(
        torch.load(_WEIGHTS_PATH, map_location="cpu")
    )
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        encoder = nn.DataParallel(encoder, device_ids=_DEVICE_IDS)
    encoder = encoder.to(_device)
    encoder.eval()
    _encoder = encoder

    # FAISS index
    _index = faiss.read_index(_INDEX_PATH)
    _id_list = np.load(_ID_MAP_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_embedding(spec_path: str) -> np.ndarray:
    """Extract an L2-normalised 512-d embedding from a spectrogram file.

    Args:
        spec_path: Path to a ``.npy`` file of shape ``(1, 128, 1292)``.

    Returns:
        Numpy array of shape ``(512,)`` with unit L2 norm.
    """
    _ensure_loaded()

    spec = np.load(spec_path).astype(np.float32)
    crop = spec[:, :, _CENTER_START:_CENTER_END].copy()
    tensor = torch.from_numpy(crop).float().unsqueeze(0).to(_device)

    with torch.no_grad():
        h = _encoder(tensor)

    h = h.cpu().numpy().flatten()
    h /= np.linalg.norm(h) + 1e-8
    return h


def predict_track(
    spec_path: str, k: int = 1
) -> Tuple[int, float]:
    """Retrieve the closest track ID for a query spectrogram.

    Args:
        spec_path: Path to a ``.npy`` spectrogram of shape ``(1, 128, 1292)``.
        k:         Number of neighbours to search (top-1 returned).

    Returns:
        Tuple of ``(predicted_track_id, cosine_similarity_score)``.
    """
    _ensure_loaded()

    query = get_embedding(spec_path).reshape(1, -1).astype(np.float32)
    D, I = _index.search(query, k)
    return int(_id_list[I[0, 0]]), float(D[0, 0])


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EchoFind — single-query demo")
    parser.add_argument("spec_path", type=str, help="Path to a .npy spectrogram.")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--id_map", type=str, default=None)
    args = parser.parse_args()

    if args.weights:
        _WEIGHTS_PATH = args.weights
    if args.index:
        _INDEX_PATH = args.index
    if args.id_map:
        _ID_MAP_PATH = args.id_map

    emb = get_embedding(args.spec_path)
    tid, sim = predict_track(args.spec_path)
    print(f"Embedding shape : {emb.shape}")
    print(f"Predicted track : {tid}")
    print(f"Cosine similarity: {sim:.4f}")
