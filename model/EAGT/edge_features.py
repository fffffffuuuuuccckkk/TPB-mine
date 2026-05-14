import torch


def normalize_input_x(x):
    """
    Normalize traffic history to [B, N, T, C].

    Supported input shapes:
    - [B, N, T, C]
    - [N, T, C]
    - [B, N, T]
    - [N, T]
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float()
    if x.dim() == 4:
        pass
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(-1)
    else:
        raise ValueError("EAGT expected x with shape [B,N,T,C], [N,T,C], [B,N,T], or [N,T], got {}".format(tuple(x.shape)))
    assert x.dim() == 4, "EAGT normalized x must be [B,N,T,C], got {}".format(tuple(x.shape))
    return x


def _node_series(x):
    """
    Convert normalized x [B, N, T, C] into node series [N, B*T].

    V1 uses channel 0 as the traffic speed/value channel.
    """
    x = normalize_input_x(x)
    B, N, T, C = x.shape
    assert C >= 1, "EAGT x channel dimension C must be >= 1"
    return x[..., 0].permute(1, 0, 2).reshape(N, B * T)


def compute_corr_matrix(x):
    """
    Compute Pearson correlation matrix from x.

    Args:
        x: traffic history [B, N, T, C] or [N, T, C].
    Returns:
        corr: FloatTensor [N, N].
    """
    s = _node_series(x)
    assert s.dim() == 2, "EAGT node series must be [N, S]"
    s = s - s.mean(dim=1, keepdim=True)
    std = s.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    z = s / std
    corr = torch.matmul(z, z.t()) / max(1, z.shape[1])
    return corr.clamp(-1.0, 1.0)


def compute_lagcorr_matrix(x, max_lag=3):
    """
    Compute max absolute lagged correlation.

    Args:
        x: traffic history [B, N, T, C] or [N, T, C].
        max_lag: non-negative integer.
    Returns:
        lagcorr_max: FloatTensor [N, N].
        lag_argmax: FloatTensor [N, N], signed best lag in [-max_lag, max_lag].
    """
    s = _node_series(x)
    N, S = s.shape
    assert S > 1, "EAGT lag correlation needs at least two time points"
    best = torch.zeros((N, N), device=s.device, dtype=s.dtype)
    best_lag = torch.zeros((N, N), device=s.device, dtype=s.dtype)
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            a, b = s[:, :lag], s[:, -lag:]
        elif lag > 0:
            a, b = s[:, lag:], s[:, :-lag]
        else:
            a, b = s, s
        if a.shape[1] < 2:
            continue
        a = (a - a.mean(dim=1, keepdim=True)) / a.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        b = (b - b.mean(dim=1, keepdim=True)) / b.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        corr = torch.matmul(a, b.t()) / max(1, a.shape[1])
        mask = corr.abs() > best.abs()
        best = torch.where(mask, corr, best)
        best_lag = torch.where(mask, torch.full_like(best_lag, float(lag)), best_lag)
    return best.clamp(-1.0, 1.0), best_lag


def build_candidate_edges(x, topk, method="corr", include_self_loop=False):
    """
    Build candidate directed node pairs by per-source-node top-k score.

    Args:
        x: traffic history [B, N, T, C] or [N, T, C].
        topk: number of outgoing candidates per node.
        method: one of ["corr", "lagcorr", "corr_lagcorr"].
        include_self_loop: whether to keep i -> i.
    Returns:
        pairs: LongTensor [E, 2].
        weights: FloatTensor [E].
    """
    assert method in ["corr", "lagcorr", "corr_lagcorr"], "unsupported EAGT candidate method {}".format(method)
    corr = compute_corr_matrix(x)
    if method == "corr":
        score = corr.abs()
    else:
        lagcorr, _ = compute_lagcorr_matrix(x)
        score = lagcorr.abs() if method == "lagcorr" else 0.5 * (corr.abs() + lagcorr.abs())

    N = score.shape[0]
    assert score.shape == (N, N), "EAGT score matrix must be [N,N], got {}".format(tuple(score.shape))
    if not include_self_loop:
        score = score.clone()
        score.fill_diagonal_(float("-inf"))
    k = min(max(1, int(topk)), N if include_self_loop else max(1, N - 1))
    vals, idx = torch.topk(score, k=k, dim=1)
    src = torch.arange(N, device=score.device).unsqueeze(1).expand_as(idx)
    pairs = torch.stack([src.reshape(-1), idx.reshape(-1)], dim=1).long()
    weights = vals.reshape(-1).float()
    keep = torch.isfinite(weights)
    pairs, weights = pairs[keep], weights[keep]
    assert pairs.dim() == 2 and pairs.shape[1] == 2, "EAGT pairs must be [E,2]"
    return pairs, weights


def edge_feature_from_pairs(x, pairs, method="corr_lagcorr"):
    """
    Compute edge features for directed pairs.

    Args:
        x: traffic history [B, N, T, C] or [N, T, C].
        pairs: LongTensor [E, 2].
        method: kept for interface compatibility; V1 always returns 8 features.
    Returns:
        feats: FloatTensor [E, 8] with
            corr, abs_corr, lagcorr_max, lag_argmax, std_i, std_j, mean_i, mean_j.
    """
    x4 = normalize_input_x(x)
    pairs = pairs.long().to(x4.device)
    assert pairs.dim() == 2 and pairs.shape[1] == 2, "EAGT pairs must be [E,2], got {}".format(tuple(pairs.shape))
    B, N, T, C = x4.shape
    assert pairs.numel() == 0 or int(pairs.max().item()) < N, "EAGT pair index out of range for N={}".format(N)

    corr = compute_corr_matrix(x4)
    lagcorr, lag_argmax = compute_lagcorr_matrix(x4)
    s = _node_series(x4)
    mean = s.mean(dim=1)
    std = s.std(dim=1, unbiased=False)

    i, j = pairs[:, 0], pairs[:, 1]
    feats = torch.stack([
        corr[i, j],
        corr[i, j].abs(),
        lagcorr[i, j],
        lag_argmax[i, j],
        std[i],
        std[j],
        mean[i],
        mean[j],
    ], dim=1).float()
    assert feats.dim() == 2 and feats.shape[0] == pairs.shape[0], "EAGT feats must be [E,F]"
    return feats

