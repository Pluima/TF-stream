import itertools

import torch
import torch.nn.functional as F


def _si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant SNR for tensors with shape [B, T]."""
    est = est - torch.mean(est, dim=-1, keepdim=True)
    ref = ref - torch.mean(ref, dim=-1, keepdim=True)

    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref
    proj = proj / (torch.sum(ref * ref, dim=-1, keepdim=True) + eps)

    noise = est - proj
    ratio = torch.sum(proj * proj, dim=-1) / (torch.sum(noise * noise, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def align_reference_for_pit(
    est_sources: torch.Tensor,
    ref_sources: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Align references to estimation order using the best-PIT assignment.

    Args:
        est_sources: [B, S, T]
        ref_sources: [B, S, T]
    Returns:
        ref_aligned: [B, S, T], where ref_aligned[:, i] matches est_sources[:, i].
    """
    if est_sources.shape != ref_sources.shape:
        raise ValueError(
            f"Shape mismatch: est={tuple(est_sources.shape)} ref={tuple(ref_sources.shape)}"
        )
    if est_sources.ndim != 3:
        raise ValueError("Expected [B, S, T] tensors")

    b, s, t = est_sources.shape
    if s == 1:
        return ref_sources

    if s == 2:
        s00 = _si_snr(est_sources[:, 0], ref_sources[:, 0], eps=eps)
        s11 = _si_snr(est_sources[:, 1], ref_sources[:, 1], eps=eps)
        s01 = _si_snr(est_sources[:, 0], ref_sources[:, 1], eps=eps)
        s10 = _si_snr(est_sources[:, 1], ref_sources[:, 0], eps=eps)
        pair_keep = 0.5 * (s00 + s11)
        pair_swap = 0.5 * (s01 + s10)
        swap_mask = (pair_swap > pair_keep).view(b, 1, 1)
        swapped = ref_sources[:, [1, 0], :]
        return torch.where(swap_mask, swapped, ref_sources)

    perms = list(itertools.permutations(range(s)))
    perm_scores = []
    for perm in perms:
        score = 0.0
        for est_i, ref_i in enumerate(perm):
            score = score + _si_snr(est_sources[:, est_i], ref_sources[:, ref_i], eps=eps)
        perm_scores.append(score / float(s))
    stacked = torch.stack(perm_scores, dim=1)  # [B, P]
    best_idx = torch.argmax(stacked, dim=1)  # [B]

    perm_tensor = torch.tensor(perms, device=est_sources.device, dtype=torch.long)  # [P,S]
    best_perm = perm_tensor[best_idx]  # [B,S]
    gather_idx = best_perm.unsqueeze(-1).expand(-1, -1, t)  # [B,S,T]
    return torch.gather(ref_sources, dim=1, index=gather_idx)


def pit_si_snr_loss(est_sources: torch.Tensor, ref_sources: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    PIT SI-SNR loss.

    Args:
        est_sources: [B, S, T]
        ref_sources: [B, S, T]
    Returns:
        scalar loss (negative SI-SNR)
    """
    if est_sources.shape != ref_sources.shape:
        raise ValueError(
            f"Shape mismatch: est={tuple(est_sources.shape)} ref={tuple(ref_sources.shape)}"
        )
    if est_sources.ndim != 3:
        raise ValueError("Expected [B, S, T] tensors")

    b, s, _ = est_sources.shape

    if s == 2:
        s00 = _si_snr(est_sources[:, 0], ref_sources[:, 0], eps=eps)
        s11 = _si_snr(est_sources[:, 1], ref_sources[:, 1], eps=eps)
        s01 = _si_snr(est_sources[:, 0], ref_sources[:, 1], eps=eps)
        s10 = _si_snr(est_sources[:, 1], ref_sources[:, 0], eps=eps)

        pair_a = (s00 + s11) * 0.5
        pair_b = (s01 + s10) * 0.5
        best = torch.maximum(pair_a, pair_b)
        return -torch.mean(best)

    perms = list(itertools.permutations(range(s)))
    perm_scores = []
    for perm in perms:
        score = 0.0
        for est_i, ref_i in enumerate(perm):
            score = score + _si_snr(est_sources[:, est_i], ref_sources[:, ref_i], eps=eps)
        perm_scores.append(score / float(s))

    stacked = torch.stack(perm_scores, dim=1)  # [B, P]
    best, _ = torch.max(stacked, dim=1)
    return -torch.mean(best)


def mixture_consistency_loss(est_sources: torch.Tensor, mix_wave: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Penalize mismatch between sum of separated sources and input mixture.

    Returns normalized MSE:
      mean_b [ mse(sum_s est[b,s], mix[b]) / (power(mix[b]) + eps) ].
    """
    if est_sources.ndim != 3:
        raise ValueError("Expected est_sources shape [B,S,T]")
    if mix_wave.ndim != 2:
        raise ValueError("Expected mix_wave shape [B,T]")
    if est_sources.shape[0] != mix_wave.shape[0]:
        raise ValueError(
            f"Batch mismatch: est={tuple(est_sources.shape)} mix={tuple(mix_wave.shape)}"
        )

    t = min(est_sources.shape[-1], mix_wave.shape[-1])
    est_sum = torch.sum(est_sources[..., :t], dim=1)  # [B,T]
    mix_t = mix_wave[..., :t]
    diff = est_sum - mix_t

    num = torch.mean(diff * diff, dim=-1)
    den = torch.mean(mix_t * mix_t, dim=-1) + eps
    return torch.mean(num / den)


def energy_ratio_loss(est_sources: torch.Tensor, ref_sources: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Match inter-source energy distribution between estimation and reference.
    """
    if est_sources.shape != ref_sources.shape:
        raise ValueError(
            f"Shape mismatch: est={tuple(est_sources.shape)} ref={tuple(ref_sources.shape)}"
        )
    if est_sources.ndim != 3:
        raise ValueError("Expected [B,S,T] tensors")

    e_est = torch.mean(est_sources * est_sources, dim=-1) + eps  # [B,S]
    e_ref = torch.mean(ref_sources * ref_sources, dim=-1) + eps  # [B,S]

    r_est = e_est / torch.sum(e_est, dim=1, keepdim=True)
    r_ref = e_ref / torch.sum(e_ref, dim=1, keepdim=True)
    return torch.mean(torch.abs(r_est - r_ref))


def source_decorrelation_loss(est_sources: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Penalize pairwise source correlation to reduce leakage overlap.
    """
    if est_sources.ndim != 3:
        raise ValueError("Expected [B,S,T] tensors")
    _, s, _ = est_sources.shape
    if s < 2:
        return torch.zeros((), device=est_sources.device, dtype=est_sources.dtype)

    x = est_sources - torch.mean(est_sources, dim=-1, keepdim=True)
    norms = torch.sqrt(torch.sum(x * x, dim=-1) + eps)  # [B,S]

    acc = 0.0
    pairs = 0
    for i in range(s):
        for j in range(i + 1, s):
            dot = torch.sum(x[:, i] * x[:, j], dim=-1)
            corr = torch.abs(dot / (norms[:, i] * norms[:, j] + eps))
            acc = acc + torch.mean(corr)
            pairs += 1

    return acc / float(max(1, pairs))


def ordered_si_snr_loss(est_sources: torch.Tensor, ref_sources: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Ordered (non-PIT) SI-SNR loss.

    Args:
        est_sources: [B, S, T]
        ref_sources: [B, S, T]
    Returns:
        scalar loss (negative ordered SI-SNR)
    """
    if est_sources.shape != ref_sources.shape:
        raise ValueError(
            f"Shape mismatch: est={tuple(est_sources.shape)} ref={tuple(ref_sources.shape)}"
        )
    if est_sources.ndim != 3:
        raise ValueError("Expected [B, S, T] tensors")

    _, s, _ = est_sources.shape
    score = 0.0
    for i in range(s):
        score = score + _si_snr(est_sources[:, i], ref_sources[:, i], eps=eps)
    score = score / float(s)
    return -torch.mean(score)


def directional_si_snr_loss(
    est_sources: torch.Tensor,
    ref_sources: torch.Tensor,
    eps: float = 1e-8,
    swap_margin_db: float = 0.3,
    swap_weight: float = 0.35,
) -> torch.Tensor:
    """
    Direction-conditioned ordered SI-SNR loss.

    For 2-speaker case, adds anti-swap ranking regularization:
    ordered score should exceed swapped score by margin.
    """
    if est_sources.shape != ref_sources.shape:
        raise ValueError(
            f"Shape mismatch: est={tuple(est_sources.shape)} ref={tuple(ref_sources.shape)}"
        )
    if est_sources.ndim != 3:
        raise ValueError("Expected [B, S, T] tensors")

    b, s, _ = est_sources.shape
    if s != 2:
        return ordered_si_snr_loss(est_sources, ref_sources, eps=eps)

    s00 = _si_snr(est_sources[:, 0], ref_sources[:, 0], eps=eps)
    s11 = _si_snr(est_sources[:, 1], ref_sources[:, 1], eps=eps)
    s01 = _si_snr(est_sources[:, 0], ref_sources[:, 1], eps=eps)
    s10 = _si_snr(est_sources[:, 1], ref_sources[:, 0], eps=eps)

    ordered = 0.5 * (s00 + s11)  # [B]
    swapped = 0.5 * (s01 + s10)  # [B]

    main = -torch.mean(ordered)
    # Enforce ordered assignment to be better than swapped by a margin.
    rank_penalty = F.relu(swapped - ordered + float(swap_margin_db))
    aux = torch.mean(rank_penalty)
    return main + float(swap_weight) * aux
