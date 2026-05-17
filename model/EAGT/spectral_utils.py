import torch
import torch.nn.functional as F

from .edge_features import compute_corr_matrix, compute_lagcorr_matrix


def normalize_square_matrix(M, fill_diag=0.0, abs_value=False):
    """Normalize a square matrix [N,N] for spectral/role use."""
    M = torch.as_tensor(M, dtype=torch.float32)
    assert M.dim() == 2 and M.shape[0] == M.shape[1], "SAGT expected square [N,N], got {}".format(tuple(M.shape))
    M = torch.nan_to_num(M.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if abs_value:
        M = M.abs()
    M = 0.5 * (M + M.t())
    if fill_diag is not None:
        M = M.clone()
        M.fill_diagonal_(float(fill_diag))
    return M


def row_normalize(M, eps=1e-8):
    """Row normalize [N,N] with zero-row protection."""
    M = torch.nan_to_num(M.float(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = M.abs().sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return M / denom


def masked_row_softmax(M, mask=None):
    """Masked row-wise softmax for [N,N] or [B,N,N]."""
    if mask is None:
        mask = M.abs() > 0
    logits = M.masked_fill(~mask, -1e9)
    out = F.softmax(logits, dim=-1) * mask.float()
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def build_relation_matrix(x, method="corr_lagcorr", abs_value=True):
    """Build target/source relation matrix [N,N] from traffic history."""
    assert method in ["corr", "lagcorr", "corr_lagcorr"], "unsupported SAGT relation method {}".format(method)
    corr = compute_corr_matrix(x)
    if method == "corr":
        M = corr
    else:
        lagcorr, _ = compute_lagcorr_matrix(x)
        M = lagcorr if method == "lagcorr" else 0.5 * (corr + lagcorr)
    return normalize_square_matrix(M, fill_diag=0.0, abs_value=abs_value)


def low_rank_reconstruct(M, rank=8, use_svd=True):
    """Low-rank reconstruct [N,N] using SVD/eigh with torch.svd fallback."""
    M = normalize_square_matrix(M, fill_diag=0.0, abs_value=False)
    N = M.shape[0]
    r = min(max(1, int(rank)), N)
    if use_svd:
        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            L = torch.matmul(U[:, :r] * S[:r].view(1, -1), Vh[:r, :])
            return L, S, U[:, :r]
        except AttributeError:
            U, S, V = torch.svd(M)
            L = torch.matmul(U[:, :r] * S[:r].view(1, -1), V[:, :r].t())
            return L, S, U[:, :r]
    try:
        vals, vecs = torch.linalg.eigh(M)
    except AttributeError:
        vals, vecs = torch.symeig(M, eigenvectors=True)
    idx = torch.argsort(vals.abs(), descending=True)[:r]
    U = vecs[:, idx]
    S = vals[idx]
    L = torch.matmul(U * S.view(1, -1), U.t())
    return L, S, U


def spectral_signature(M, rank=16, moments=4):
    """Permutation-insensitive spectral signature with shape [rank + moments]."""
    M = normalize_square_matrix(M, fill_diag=0.0, abs_value=True)
    M = row_normalize(M.abs())
    try:
        vals = torch.linalg.eigvalsh(0.5 * (M + M.t()))
    except AttributeError:
        vals = torch.symeig(0.5 * (M + M.t()), eigenvectors=False)[0]
    vals = torch.sort(vals.abs(), descending=True).values
    r = int(rank)
    if vals.numel() < r:
        vals_pad = F.pad(vals, (0, r - vals.numel()))
    else:
        vals_pad = vals[:r]
    moment_vals = []
    for p in range(1, int(moments) + 1):
        moment_vals.append(torch.mean(vals.pow(p)))
    return torch.cat([vals_pad, torch.stack(moment_vals)], dim=0).float()


def sym_nmf_torch(M, rank=8, iters=80, lr=0.05, eps=1e-8, nonnegative=True):
    """Factorize M [N,N] as W [N,K] B [K,K] W^T using lightweight projected Adam."""
    M = normalize_square_matrix(M, fill_diag=0.0, abs_value=True).detach()
    N = M.shape[0]
    K = min(max(1, int(rank)), N)
    with torch.no_grad():
        _, _, U = low_rank_reconstruct(M, rank=K, use_svd=False)
        W0 = U.abs() if nonnegative else U
        if W0.shape[1] < K:
            W0 = F.pad(W0, (0, K - W0.shape[1]))
        B0 = torch.eye(K, device=M.device, dtype=M.dtype)
    W = W0.clone().detach().requires_grad_(True)
    B = B0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([W, B], lr=float(lr))
    last_loss = None
    for _ in range(max(1, int(iters))):
        opt.zero_grad()
        B_sym = 0.5 * (B + B.t())
        recon = torch.matmul(torch.matmul(W, B_sym), W.t())
        loss = (recon - M).pow(2).mean()
        loss.backward()
        opt.step()
        if nonnegative:
            with torch.no_grad():
                W.clamp_(min=float(eps))
                B.clamp_(min=float(eps))
        last_loss = loss.detach()
    with torch.no_grad():
        B_final = 0.5 * (B + B.t())
        recon = torch.matmul(torch.matmul(W, B_final), W.t())
    return W.detach(), B_final.detach(), recon.detach(), last_loss if last_loss is not None else torch.zeros((), device=M.device)


def safe_topk_row(A, k):
    """Keep row-wise top-k entries for [N,N] or [B,N,N]."""
    if k <= 0 or k >= A.shape[-1]:
        return A
    vals, idx = torch.topk(A, k=int(k), dim=-1)
    out = torch.zeros_like(A)
    out.scatter_(-1, idx, vals)
    return out


def safe_normalize_score(x, eps=1e-8):
    """Min-max normalize a score vector to [0,1], fallback to zeros when flat."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    span = hi - lo
    if float(span.detach().abs().item()) < eps:
        return torch.zeros_like(x)
    return (x - lo) / span.clamp_min(float(eps))
