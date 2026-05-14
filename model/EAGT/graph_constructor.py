import torch
import torch.nn as nn
import torch.nn.functional as F

from .edge_features import build_candidate_edges, edge_feature_from_pairs
from .evidence_retriever import EvidenceRetriever
from .debug_utils import summarize_adj


class EAGTGraphConstructor(nn.Module):
    """
    Evidence-Attributed Graph Transfer V1.

    Input:
        x_target: traffic history [B,N,T,C] or [N,T,C].
        A_original: optional original adjacency [N,N] or [B,N,N].
        source_cache: SourceEvidenceCache with features [E_s,F].
    Output:
        A_final: FloatTensor [B,N,N] if A_original is batched else [N,N].
        aux_loss_dict: sparse/evidence losses, both scalar tensors.
        debug_dict: target pairs, retrieval attribution, and adjacency stats.
    """
    def __init__(self, args):
        super(EAGTGraphConstructor, self).__init__()
        self.mode = _get_arg(args, "eagt_mode", "edge_v1")
        assert self.mode in ["edge_v1", "edge_subgraph_v2", "full_v3"], "Unsupported eagt_mode {}".format(self.mode)
        self.candidate_topk = int(_get_arg(args, "eagt_candidate_topk", 20))
        self.candidate_method = _get_arg(args, "eagt_candidate_method", "corr")
        self.include_self_loop = bool(_get_arg(args, "eagt_include_self_loop", 0))
        self.retrieval_topk = int(_get_arg(args, "eagt_retrieval_topk", 8))
        self.chunk_size = int(_get_arg(args, "eagt_chunk_size", 4096))
        self.gamma = float(_get_arg(args, "eagt_gamma", 0.0))
        self.row_softmax = bool(_get_arg(args, "eagt_row_softmax", 1))
        self.sparse_topk = int(_get_arg(args, "eagt_sparse_topk", 20))
        self.random_evidence = bool(_get_arg(args, "eagt_random_evidence", 0))
        self.retriever = EvidenceRetriever(
            w_importance=float(_get_arg(args, "eagt_w_importance", 0.1)),
            w_grad=float(_get_arg(args, "eagt_w_grad", 0.0)),
        )

    def forward(self, x_target, A_original=None, source_cache=None, return_debug=False):
        if self.mode != "edge_v1":
            raise NotImplementedError("EAGT currently implements only edge_v1")
        assert source_cache is not None, "EAGT source_cache is required when use_eagt=True"
        source_feats = source_cache.get_features()
        source_meta = source_cache.get_metadata()
        assert source_feats.dim() == 2 and source_feats.shape[0] == len(source_meta), "bad EAGT source cache"

        pairs_t, weights_t = build_candidate_edges(
            x_target,
            topk=self.candidate_topk,
            method=self.candidate_method,
            include_self_loop=self.include_self_loop,
        )
        feats_t = edge_feature_from_pairs(x_target, pairs_t, method="corr_lagcorr")
        device = feats_t.device
        source_feats = source_feats.to(device=device, dtype=feats_t.dtype)
        source_importance = torch.tensor(
            [m.get("source_importance", 0.0) for m in source_meta],
            device=device,
            dtype=feats_t.dtype,
        )
        topk_idx, topk_score, topk_alpha = self.retriever(
            feats_t,
            source_feats,
            source_importance=source_importance,
            topk=self.retrieval_topk,
            chunk_size=self.chunk_size,
            random=self.random_evidence,
        )
        edge_score = (topk_alpha * source_importance[topk_idx]).sum(dim=1)
        N = _infer_n(x_target, A_original)
        A_eagt = torch.zeros((N, N), device=device, dtype=feats_t.dtype)
        A_eagt[pairs_t[:, 0], pairs_t[:, 1]] = edge_score
        if self.row_softmax:
            A_eagt = _masked_row_softmax(A_eagt)
        if self.sparse_topk > 0:
            A_eagt = _row_topk(A_eagt, self.sparse_topk)
        A_final = self._fuse(A_original, A_eagt)
        zero = torch.zeros((), device=device, dtype=feats_t.dtype)
        aux_loss_dict = {
            "eagt_sparse_loss": A_eagt.abs().mean(),
            "eagt_evidence_loss": zero,
        }
        debug_dict = {
            "target_pairs": pairs_t.detach(),
            "target_edge_weight": weights_t.detach(),
            "topk_source_idx": topk_idx.detach(),
            "topk_score": topk_score.detach(),
            "topk_alpha": topk_alpha.detach(),
            "A_original_stats": summarize_adj(A_original, "A_original"),
            "A_eagt_stats": summarize_adj(A_eagt, "A_eagt"),
            "A_final_stats": summarize_adj(A_final, "A_final"),
            "target_candidate_count": int(pairs_t.shape[0]),
        } if return_debug else {}
        return A_final, aux_loss_dict, debug_dict

    def _fuse(self, A_original, A_eagt):
        if A_original is None:
            return A_eagt
        A_original = A_original.to(device=A_eagt.device, dtype=A_eagt.dtype)
        assert A_original.dim() in [2, 3], "A_original must be [N,N] or [B,N,N], got {}".format(tuple(A_original.shape))
        if self.gamma == 0.0:
            return A_original
        if A_original.dim() == 3:
            A_eagt_use = A_eagt.unsqueeze(0).expand(A_original.shape[0], -1, -1)
        else:
            A_eagt_use = A_eagt
        return (1.0 - self.gamma) * A_original + self.gamma * A_eagt_use


def _masked_row_softmax(A):
    mask = A.abs() > 0
    logits = A.masked_fill(~mask, -1e9)
    out = F.softmax(logits, dim=-1)
    out = out * mask.float()
    denom = out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out / denom


def _row_topk(A, topk):
    if topk <= 0 or topk >= A.shape[-1]:
        return A
    vals, idx = torch.topk(A, k=int(topk), dim=-1)
    out = torch.zeros_like(A)
    out.scatter_(-1, idx, vals)
    return out


def _infer_n(x_target, A_original):
    if A_original is not None:
        return A_original.shape[-1]
    x = torch.as_tensor(x_target)
    if x.dim() == 4:
        return x.shape[1]
    if x.dim() == 3:
        return x.shape[0]
    raise ValueError("Cannot infer N for EAGT from x shape {}".format(tuple(x.shape)))


def _get_arg(args, name, default):
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)

