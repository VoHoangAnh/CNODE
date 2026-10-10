"""Loss terms of CNODE (Eqs. 3, 4, 10, 11)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Eq. (10): squared L2 error summed over features, averaged over visits in ``mask``."""
    sq = ((x - x_hat) ** 2).sum(-1)               # (B, K)
    return (sq * mask).sum() / mask.sum().clamp_min(1)


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Eq. (11): KL(N(mu, diag(sigma^2)) || N(0, I)), summed over dims, averaged over subjects."""
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1)
    return kl.mean()


def metadata_weighted_infonce(
    x: torch.Tensor,
    t_aligned: torch.Tensor,
    cond: torch.Tensor,
    subject_ids: torch.Tensor,
    delta: float = 0.5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Eqs. (3)-(4): metadata-weighted InfoNCE on aligned visit times.

    Args:
        x:            (N, D) shape features of all visits in the batch (anchors and candidates).
        t_aligned:    (N,)   aligned times ``t~`` (receives gradients -> tau, gamma).
        cond:         (N, Dc) subject conditions [M_i, C_i] repeated per visit.
        subject_ids:  (N,)   integer subject index for each visit.
        delta:        cosine-similarity threshold on shape features defining positives.
        temperature:  softmax temperature (the paper reuses the symbol tau for it).

    Implementation choices where the paper is under-specified (documented in README):
      * ``s(t~_i, t~_j)``: the paper states "cosine similarity between aligned times", which is
        degenerate for scalars (always +-1). We use a negative squared distance,
        ``s = -(t~_i - t~_j)^2``, so similar visits are pulled to nearby aligned times.
      * Pairs from the same subject are never positives (otherwise the alignment collapses
        gamma -> 0); they remain in the denominator, i.e. act as negatives.
      * ``w_ij = (1 + cos([M_i, C_i], [M_j, C_j])) / 2`` maps the condition similarity to [0, 1].
      * The loss is averaged over positives per anchor and over anchors that have positives.
    """
    n = x.shape[0]
    if n < 2:
        return t_aligned.new_zeros(())
    eye = torch.eye(n, dtype=torch.bool, device=x.device)
    same_subject = subject_ids[:, None] == subject_ids[None, :]

    with torch.no_grad():
        feat_sim = F.cosine_similarity(x[:, None, :], x[None, :, :], dim=-1)
        positive = (feat_sim > delta) & ~same_subject
        cond_sim = F.cosine_similarity(cond[:, None, :], cond[None, :, :], dim=-1)
        weight = 0.5 * (1.0 + cond_sim)

    logits = -(t_aligned[:, None] - t_aligned[None, :]).pow(2) / temperature
    logits = logits.masked_fill(eye, float("-inf"))          # exclude the anchor itself
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = log_prob.masked_fill(eye, 0.0)

    pos_w = weight * positive
    n_pos = positive.sum(1)
    has_pos = n_pos > 0
    if not has_pos.any():
        return t_aligned.new_zeros(())
    per_anchor = -(pos_w * log_prob).sum(1) / n_pos.clamp_min(1)
    return per_anchor[has_pos].mean()
