import torch
import torch.nn as nn

from .edge_features import build_candidate_edges, edge_feature_from_pairs, normalize_input_x
from .evidence_retriever import EvidenceRetriever
from .spectral_utils import (
    build_relation_matrix,
    low_rank_reconstruct,
    masked_row_softmax,
    normalize_square_matrix,
    safe_normalize_score,
    safe_topk_row,
    spectral_signature,
    sym_nmf_torch,
)


class SAGTGraphConstructor(nn.Module):
    """
    Spectral Source-Attributed Graph Transfer.

    Input:
        x_target: Tensor [B,N,T,C] or [N,T,C].
        A_original: optional TPB graph [B,N,N] or [N,N].
        source_structure_cache: SourceStructureCache.
        source_evidence_cache: optional SourceEvidenceCache for EAGT score.
    Output:
        A_final: Tensor [B,N,N] when A_original is batched, otherwise [N,N].
        aux_loss_dict: scalar SAGT losses.
        debug_dict: compact attribution tensors when return_debug=True.
    """
    def __init__(self, args):
        super(SAGTGraphConstructor, self).__init__()
        self.use_lowrank = _as_bool(_get_arg(args, "sagt_use_lowrank", 1))
        self.lowrank_rank = int(_get_arg(args, "sagt_lowrank_rank", 8))
        self.lowrank_source = _get_arg(args, "sagt_lowrank_source", "corr_lagcorr")
        self.lowrank_abs = _as_bool(_get_arg(args, "sagt_lowrank_abs", 1))
        self.lowrank_row_softmax = _as_bool(_get_arg(args, "sagt_lowrank_row_softmax", 1))
        self.use_source_roles = _as_bool(_get_arg(args, "sagt_use_source_roles", 1))
        self.role_dim = int(_get_arg(args, "sagt_role_dim", 8))
        self.role_iters = int(_get_arg(args, "sagt_role_iters", 80))
        self.role_lr = float(_get_arg(args, "sagt_role_lr", 0.05))
        self.role_eps = float(_get_arg(args, "sagt_role_eps", 1e-8))
        self.role_nonnegative = _as_bool(_get_arg(args, "sagt_role_nonnegative", 1))
        self.use_spectral_signature = _as_bool(_get_arg(args, "sagt_use_spectral_signature", 1))
        self.spectral_rank = int(_get_arg(args, "sagt_spectral_rank", 16))
        self.spectral_moments = int(_get_arg(args, "sagt_spectral_moments", 4))
        self.spectral_tau = max(float(_get_arg(args, "sagt_spectral_tau", 1.0)), 1e-8)
        self.alpha = float(_get_arg(args, "sagt_alpha_lowrank", 0.3))
        self.beta = float(_get_arg(args, "sagt_beta_src_role", 0.4))
        self.gamma = float(_get_arg(args, "sagt_gamma_eagt", 0.2))
        self.delta = float(_get_arg(args, "sagt_delta_res", 0.1))
        self.sparse_topk = int(_get_arg(args, "sagt_sparse_topk", 20))
        self.candidate_topk = int(_get_arg(args, "eagt_candidate_topk", self.sparse_topk))
        self.include_self_loop = _as_bool(_get_arg(args, "eagt_include_self_loop", 0))
        self.use_eagt = _as_bool(_get_arg(args, "use_eagt", 0))
        self.eagt_retrieval_topk = int(_get_arg(args, "eagt_retrieval_topk", 8))
        self.eagt_chunk_size = int(_get_arg(args, "eagt_chunk_size", 4096))
        self.eagt_random_evidence = _as_bool(_get_arg(args, "eagt_random_evidence", 0))
        self.retriever = EvidenceRetriever(
            w_importance=float(_get_arg(args, "eagt_w_importance", 0.1)),
            w_grad=float(_get_arg(args, "eagt_w_grad", 0.0)),
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_target, A_original=None, source_structure_cache=None,
                source_evidence_cache=None, return_debug=False):
        if source_structure_cache is None:
            raise ValueError("SAGT requires source_structure_cache when use_sagt=True.")
        x4 = normalize_input_x(x_target)
        B, N, T, C = x4.shape
        assert C >= 1, "SAGT expected x_target [B,N,T,C] with C>=1, got {}".format(tuple(x4.shape))
        pairs, target_edge_weight = build_candidate_edges(
            x4,
            topk=self.candidate_topk,
            method=self.lowrank_source,
            include_self_loop=self.include_self_loop,
        )
        pairs = pairs.to(x4.device)
        target_edge_weight = target_edge_weight.to(x4.device, dtype=x4.dtype)
        assert pairs.dim() == 2 and pairs.shape[1] == 2, "SAGT pairs must be [E,2], got {}".format(tuple(pairs.shape))

        with torch.no_grad():
            M0 = build_relation_matrix(x4, method=self.lowrank_source, abs_value=self.lowrank_abs).to(x4.device)
            A_lowrank = self._lowrank_matrix(M0)
            score_lowrank = A_lowrank[pairs[:, 0], pairs[:, 1]]
            rho = self._source_weights(M0, source_structure_cache)
        W_t, _, _, _ = sym_nmf_torch(
            M0,
            rank=self.role_dim,
            iters=self.role_iters,
            lr=self.role_lr,
            eps=self.role_eps,
            nonnegative=self.role_nonnegative,
        )
        with torch.no_grad():
            A_src_role = self._source_role_matrix(W_t, source_structure_cache, rho)
            score_src_role = A_src_role[pairs[:, 0], pairs[:, 1]]
            score_eagt, A_eagt = self._eagt_scores(x4, pairs, source_evidence_cache)

        feats_t = edge_feature_from_pairs(x4, pairs, method="corr_lagcorr").to(x4.device)
        score_res = self.residual_mlp(feats_t).squeeze(-1)
        A_res = torch.zeros((N, N), device=x4.device, dtype=x4.dtype)
        A_res[pairs[:, 0], pairs[:, 1]] = safe_normalize_score(score_res).to(x4.dtype)

        score_lowrank_n = safe_normalize_score(score_lowrank).to(x4.dtype) if self.use_lowrank else torch.zeros_like(score_res)
        score_src_role_n = safe_normalize_score(score_src_role).to(x4.dtype) if self.use_source_roles else torch.zeros_like(score_res)
        score_eagt_n = safe_normalize_score(score_eagt).to(x4.dtype) if self.use_eagt else torch.zeros_like(score_res)
        score_res_n = safe_normalize_score(score_res).to(x4.dtype)
        score_final = (
            self.alpha * score_lowrank_n +
            self.beta * score_src_role_n +
            self.gamma * score_eagt_n +
            self.delta * score_res_n
        )
        A_sagt = torch.zeros((N, N), device=x4.device, dtype=x4.dtype)
        A_sagt[pairs[:, 0], pairs[:, 1]] = torch.sigmoid(score_final)
        A_sagt = masked_row_softmax(A_sagt)
        A_sagt = safe_topk_row(A_sagt, self.sparse_topk)
        A_final = _match_original_batch(A_sagt, A_original)
        zero = torch.zeros((), device=x4.device, dtype=x4.dtype)
        aux_loss_dict = {
            "sagt_sparse_loss": A_sagt.abs().mean(),
            "sagt_rank_loss": (A_sagt - A_lowrank.to(A_sagt.dtype)).pow(2).mean() if A_lowrank.shape == A_sagt.shape else zero,
            "sagt_res_loss": (score_res.abs() * score_src_role_n.detach()).mean(),
            "sagt_spec_loss": zero,
        }
        debug_dict = self._debug_dict(
            pairs, target_edge_weight, A_original, A_lowrank, A_src_role, A_eagt,
            A_res, A_sagt, A_final, score_lowrank_n, score_src_role_n,
            score_eagt_n, score_res_n, rho
        ) if return_debug else {}
        return A_final, aux_loss_dict, debug_dict

    def _lowrank_matrix(self, M0):
        if not self.use_lowrank:
            return torch.zeros_like(M0)
        L, _, _ = low_rank_reconstruct(M0, rank=self.lowrank_rank)
        L = normalize_square_matrix(L, fill_diag=0.0, abs_value=True)
        return masked_row_softmax(L) if self.lowrank_row_softmax else L

    def _source_weights(self, M0, cache):
        signatures = cache.get_spectral_signatures()
        assert signatures is not None and signatures.dim() == 2, "SAGT source spectral signatures must be [M,D]"
        signatures = signatures.to(M0.device, dtype=M0.dtype)
        if signatures.shape[0] == 1 or not self.use_spectral_signature:
            return torch.ones((signatures.shape[0],), device=M0.device, dtype=M0.dtype) / max(1, signatures.shape[0])
        sig_t = spectral_signature(M0, rank=self.spectral_rank, moments=self.spectral_moments).to(M0.device, dtype=M0.dtype)
        assert sig_t.shape[-1] == signatures.shape[-1], (
            "SAGT spectral signature dim mismatch: target {}, source {}. Rebuild cache."
            .format(tuple(sig_t.shape), tuple(signatures.shape))
        )
        dist = (signatures - sig_t.view(1, -1)).pow(2).sum(dim=-1)
        return torch.softmax(-dist / self.spectral_tau, dim=0)

    def _source_role_matrix(self, W_t, cache, rho):
        B_list = cache.get_role_B()
        assert len(B_list) == rho.shape[0], "SAGT role_B count {} != rho {}".format(len(B_list), tuple(rho.shape))
        out = torch.zeros((W_t.shape[0], W_t.shape[0]), device=W_t.device, dtype=W_t.dtype)
        for idx, B_s in enumerate(B_list):
            B_s = B_s.to(device=W_t.device, dtype=W_t.dtype)
            if B_s.shape[0] != self.role_dim:
                raise ValueError("SAGT role_dim mismatch: cache B {}, current {}. Rebuild cache.".format(tuple(B_s.shape), self.role_dim))
            out = out + rho[idx] * torch.matmul(torch.matmul(W_t, B_s), W_t.t())
        return normalize_square_matrix(out, fill_diag=0.0, abs_value=True)

    def _eagt_scores(self, x4, pairs, source_evidence_cache):
        N = x4.shape[1]
        if (not self.use_eagt) or source_evidence_cache is None:
            return torch.zeros((pairs.shape[0],), device=x4.device, dtype=x4.dtype), torch.zeros((N, N), device=x4.device, dtype=x4.dtype)
        source_feats = source_evidence_cache.get_features().to(x4.device)
        source_meta = source_evidence_cache.get_metadata()
        feats_t = edge_feature_from_pairs(x4, pairs, method="corr_lagcorr").to(x4.device)
        importance = torch.tensor(
            [m.get("source_importance", 0.0) for m in source_meta],
            device=x4.device,
            dtype=feats_t.dtype,
        )
        idx, _, alpha = self.retriever(
            feats_t,
            source_feats.to(dtype=feats_t.dtype),
            source_importance=importance,
            topk=self.eagt_retrieval_topk,
            chunk_size=self.eagt_chunk_size,
            random=self.eagt_random_evidence,
        )
        score = (alpha * importance[idx]).sum(dim=1)
        A = torch.zeros((N, N), device=x4.device, dtype=x4.dtype)
        A[pairs[:, 0], pairs[:, 1]] = safe_normalize_score(score).to(x4.dtype)
        return score, A

    def _debug_dict(self, pairs, target_edge_weight, A_original, A_lowrank, A_src_role,
                    A_eagt, A_res, A_sagt, A_final, score_lowrank, score_src_role,
                    score_eagt, score_res, rho):
        eps = 1e-8
        source_ratio = score_src_role.abs() / (score_src_role.abs() + score_res.abs() + eps)
        residual_ratio = score_res.abs() / (score_src_role.abs() + score_res.abs() + eps)
        top_city = int(torch.argmax(rho).detach().cpu().item()) if rho.numel() > 0 else -1
        return {
            "target_pairs": pairs.detach(),
            "target_edge_weight": target_edge_weight.detach(),
            "A_lowrank_stats": _stats(A_lowrank, "A_lowrank"),
            "A_src_role_stats": _stats(A_src_role, "A_src_role"),
            "A_eagt_stats": _stats(A_eagt, "A_eagt"),
            "A_res_stats": _stats(A_res, "A_res"),
            "A_sagt_stats": _stats(A_sagt, "A_sagt"),
            "A_final_stats": _stats(A_final, "A_final"),
            "score_lowrank": score_lowrank.detach(),
            "score_src_role": score_src_role.detach(),
            "score_eagt": score_eagt.detach(),
            "score_res": score_res.detach(),
            "source_ratio": source_ratio.detach(),
            "residual_ratio": residual_ratio.detach(),
            "source_city_weight": rho.detach(),
            "top_source_city_idx": torch.full((pairs.shape[0],), top_city, device=pairs.device, dtype=torch.long),
            "A_final_weight": _gather_edge_values(A_final, pairs),
            "target_candidate_count": int(pairs.shape[0]),
        }


def _match_original_batch(A_sagt, A_original):
    if A_original is None:
        return A_sagt
    A_original = A_original.to(device=A_sagt.device, dtype=A_sagt.dtype)
    assert A_original.dim() in [2, 3], "SAGT A_original must be [N,N] or [B,N,N], got {}".format(tuple(A_original.shape))
    if A_original.dim() == 3:
        return A_sagt.unsqueeze(0).expand(A_original.shape[0], -1, -1)
    return A_sagt


def _stats(A, name):
    if A is None:
        return {"name": name, "min": 0.0, "max": 0.0, "mean": 0.0, "sparsity": 1.0}
    A = torch.nan_to_num(A.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "name": name,
        "min": float(A.min().item()),
        "max": float(A.max().item()),
        "mean": float(A.mean().item()),
        "sparsity": float((A.abs() <= 1e-12).float().mean().item()),
    }


def _gather_edge_values(A, pairs):
    if A is None:
        return torch.zeros((pairs.shape[0],), device=pairs.device, dtype=torch.float32)
    if A.dim() == 3:
        return A[:, pairs[:, 0], pairs[:, 1]].mean(dim=0).detach()
    return A[pairs[:, 0], pairs[:, 1]].detach()


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
