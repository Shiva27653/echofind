"""
EchoFind — NT-Xent (Normalized Temperature-scaled Cross-Entropy) Loss

Implements the contrastive loss from SimCLR (Chen et al., 2020) over
pairs of augmented views projected into a 128-d embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent contrastive loss for SimCLR.

    Given a batch of *N* pairs, the input tensor ``z`` has shape ``(2N, D)``
    where the first *N* rows are view-1 embeddings and the last *N* rows are
    view-2 embeddings.  For each anchor ``i``, the positive is at index
    ``i + N`` (and vice-versa).  All other ``2N − 2`` samples are negatives.

    The pairwise cosine-similarity matrix is scaled by ``1 / temperature``
    before a standard cross-entropy is applied.

    Args:
        temperature: Softmax temperature τ (default: 0.1).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the NT-Xent loss.

        Args:
            z: Projected embeddings of shape ``(2N, D)``, concatenated as
               ``[view1_embeddings ; view2_embeddings]``.

        Returns:
            Scalar loss tensor.
        """
        n2 = z.shape[0]
        n = n2 // 2

        # L2-normalize → cosine similarity via dot product
        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out self-similarities (diagonal)
        mask = torch.eye(n2, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive-pair labels: view1[i] ↔ view2[i]
        labels = torch.cat(
            [torch.arange(n, n2), torch.arange(0, n)]
        ).to(z.device)

        return F.cross_entropy(sim, labels)
