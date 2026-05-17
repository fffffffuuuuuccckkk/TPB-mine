from pathlib import Path

import torch

from .spectral_utils import (
    build_relation_matrix,
    normalize_square_matrix,
    spectral_signature,
    sym_nmf_torch,
)


class SourceStructureCache(object):
    """Source spectral-role cache for SAGT."""

    def __init__(self, cache_dir="./save/sagt_cache", device="cpu"):
        self.cache_dir = cache_dir
        self.device = torch.device(device)
        self.role_B_list = []
        self.role_W_stats_list = []
        self.spectral_signatures = None
        self.city_names = []
        self.source_matrices_stats = []
        self.exemplar_edges = {}

    def build_from_source_data(self, source_data_dict, source_adj_dict=None, args=None):
        role_dim = int(_get_arg(args, "sagt_role_dim", 8))
        role_iters = int(_get_arg(args, "sagt_role_iters", 80))
        role_lr = float(_get_arg(args, "sagt_role_lr", 0.05))
        role_eps = float(_get_arg(args, "sagt_role_eps", 1e-8))
        role_max_nodes = int(_get_arg(args, "sagt_role_max_nodes", 800))
        role_source_matrix = _get_arg(args, "sagt_role_source_matrix", "adj_corr")
        role_nonnegative = _as_bool(_get_arg(args, "sagt_role_nonnegative", 1))
        sig_rank = int(_get_arg(args, "sagt_spectral_rank", 16))
        sig_moments = int(_get_arg(args, "sagt_spectral_moments", 4))
        lowrank_source = _get_arg(args, "sagt_lowrank_source", "corr_lagcorr")
        lowrank_abs = _as_bool(_get_arg(args, "sagt_lowrank_abs", 1))

        self.role_B_list, self.role_W_stats_list = [], []
        self.city_names, self.source_matrices_stats = [], []
        self.exemplar_edges = {}
        signatures = []
        source_adj_dict = source_adj_dict or {}
        for city, x in source_data_dict.items():
            M_corr = build_relation_matrix(x, method=lowrank_source, abs_value=lowrank_abs).cpu()
            adj = source_adj_dict.get(city, None)
            if adj is not None:
                M_adj = normalize_square_matrix(torch.as_tensor(adj, dtype=torch.float32), fill_diag=0.0, abs_value=True).cpu()
            else:
                M_adj = torch.zeros_like(M_corr)
            if role_source_matrix == "adj":
                M_role = M_adj
            elif role_source_matrix == "corr":
                M_role = M_corr
            else:
                M_role = 0.5 * M_adj + 0.5 * M_corr if adj is not None else M_corr
            M_role = normalize_square_matrix(M_role, fill_diag=0.0, abs_value=True)
            full_N = M_role.shape[0]
            if full_N > role_max_nodes:
                idx = torch.linspace(0, full_N - 1, steps=role_max_nodes).long()
                M_factor = M_role[idx][:, idx]
                print("[SAGT] role matrix for {} sampled from N={} to N={}".format(city, full_N, role_max_nodes))
            else:
                idx = None
                M_factor = M_role
            sig = spectral_signature(M_corr, rank=sig_rank, moments=sig_moments)
            W, B, recon, loss = sym_nmf_torch(
                M_factor,
                rank=role_dim,
                iters=role_iters,
                lr=role_lr,
                eps=role_eps,
                nonnegative=role_nonnegative,
            )
            self.city_names.append(str(city))
            self.role_B_list.append(B.cpu())
            self.role_W_stats_list.append({
                "mean": W.mean(dim=0).cpu(),
                "std": W.std(dim=0, unbiased=False).cpu(),
                "loss": float(loss.detach().cpu().item()),
                "node_count": int(full_N),
            })
            signatures.append(sig.cpu())
            self.source_matrices_stats.append(_matrix_stats(M_role, city))
            self.exemplar_edges[str(city)] = _build_exemplars(M_factor, W, B, city, max_edges=100)
        self.spectral_signatures = torch.stack(signatures, dim=0) if signatures else torch.empty(0, 0)
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "role_B_list": self.role_B_list,
            "role_W_stats_list": self.role_W_stats_list,
            "spectral_signatures": self.spectral_signatures,
            "city_names": self.city_names,
            "source_matrices_stats": self.source_matrices_stats,
            "exemplar_edges": self.exemplar_edges,
        }, path)
        return path

    def load(self, path):
        data = torch.load(path, map_location="cpu")
        self.role_B_list = data["role_B_list"]
        self.role_W_stats_list = data.get("role_W_stats_list", [])
        self.spectral_signatures = data["spectral_signatures"]
        self.city_names = data["city_names"]
        self.source_matrices_stats = data.get("source_matrices_stats", [])
        self.exemplar_edges = data.get("exemplar_edges", {})
        return self

    def to(self, device):
        self.device = torch.device(device)
        self.role_B_list = [B.to(self.device) for B in self.role_B_list]
        if self.spectral_signatures is not None:
            self.spectral_signatures = self.spectral_signatures.to(self.device)
        return self

    def get_role_B(self):
        return self.role_B_list

    def get_spectral_signatures(self):
        return self.spectral_signatures

    def get_city_names(self):
        return self.city_names

    def get_exemplars(self):
        return self.exemplar_edges


def _build_exemplars(M, W, B, city, max_edges=100):
    score = torch.matmul(torch.matmul(W, B), W.t())
    mask = M > 0
    if mask.sum().item() == 0:
        mask = ~torch.eye(M.shape[0], dtype=torch.bool, device=M.device)
    vals = score.masked_fill(~mask, -1e9).reshape(-1)
    k = min(int(max_edges), vals.numel())
    top_vals, top_idx = torch.topk(vals, k=k)
    rows = []
    N = M.shape[0]
    for value, flat_idx in zip(top_vals.cpu().tolist(), top_idx.cpu().tolist()):
        if value <= -1e8:
            continue
        u, v = divmod(int(flat_idx), N)
        rows.append({
            "source_city": str(city),
            "source_u": int(u),
            "source_v": int(v),
            "source_role_score": float(value),
            "source_adj_weight": float(M[u, v].detach().cpu().item()),
        })
    return rows


def _matrix_stats(M, city):
    return {
        "source_city": str(city),
        "node_count": int(M.shape[0]),
        "min": float(M.min().item()),
        "max": float(M.max().item()),
        "mean": float(M.mean().item()),
        "sparsity": float((M.abs() <= 1e-12).float().mean().item()),
    }


def _get_arg(args, name, default):
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ["1", "true", "yes", "y", "on"]
    return bool(value)
