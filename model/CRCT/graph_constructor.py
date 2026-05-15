import torch
import torch.nn as nn
import torch.nn.functional as F

from EAGT.edge_features import build_candidate_edges, normalize_input_x

from .debug_utils import summarize_adj
from .losses import balance_loss, entropy_loss, sparse_loss
from .relation_encoder import NodeTemporalEncoder, PairRelationEncoder
from .relation_heads import KnownRelationHeads, KnownnessHead, UnknownEdgeBranch
from .sparse_ops import as_bool


class CRCTGraphConstructor(nn.Module):
    """
    Class-Relation Concept Transfer V1.

    Input:
        x_target: historical target sequence [B,N,T,C] or [N,T,C].
        A_original: original TPB adjacency [N,N] or [B,N,N].
    Output:
        A_final: fused adjacency, same batch style as A_original when provided.
        aux_loss_dict: scalar CRCT losses on x_target.device.
        debug_dict: compact relation attribution tensors when return_debug=True.
    """
    def __init__(self, args):
        super(CRCTGraphConstructor, self).__init__()
        self.mode = _get_arg(args, "crct_mode", "v1")
        assert self.mode in ["v1", "v2_relation_kd", "v3_concept"], "Unsupported crct_mode {}".format(self.mode)
        self.candidate_topk = int(_get_arg(args, "crct_candidate_topk", 20))
        self.candidate_method = _get_arg(args, "crct_candidate_method", "corr")
        assert self.candidate_method in ["corr", "lagcorr", "corr_lagcorr", "dense"], "unsupported crct_candidate_method {}".format(self.candidate_method)
        self.include_self_loop = as_bool(_get_arg(args, "crct_include_self_loop", 0))
        self.sparse_topk = int(_get_arg(args, "crct_sparse_topk", 20))
        self.row_softmax = as_bool(_get_arg(args, "crct_row_softmax", 1))
        self.rho = float(_get_arg(args, "crct_rho", 0.0))
        self.num_relations = int(_get_arg(args, "crct_num_relations", 8))
        self.unknown_floor = float(_get_arg(args, "crct_unknown_floor", 0.0))
        self.use_unknown = as_bool(_get_arg(args, "crct_use_unknown", 1))

        hidden_dim = int(_get_arg(args, "crct_hidden_dim", 64))
        relation_dim = int(_get_arg(args, "crct_relation_dim", 64))
        dropout = float(_get_arg(args, "crct_dropout", 0.1))
        self.node_encoder = NodeTemporalEncoder(
            encoder_type=_get_arg(args, "crct_node_encoder", "tcn"),
            input_channels=int(_get_arg(args, "crct_input_channels", 1)),
            window_size=int(_get_arg(args, "his_num", 288)),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.pair_encoder = PairRelationEncoder(
            node_dim=hidden_dim,
            relation_dim=relation_dim,
            dropout=dropout,
        )
        self.known_heads = KnownRelationHeads(
            relation_dim=relation_dim,
            num_relations=self.num_relations,
            attribution=_get_arg(args, "crct_attribution", "sparsemax"),
            temperature=float(_get_arg(args, "crct_temperature", 1.0)),
        )
        self.knownness_head = KnownnessHead(
            relation_dim=relation_dim,
            num_relations=self.num_relations,
            method=_get_arg(args, "crct_knownness_method", "entropy"),
        )
        self.unknown_branch = UnknownEdgeBranch(relation_dim=relation_dim, dropout=dropout)

    def forward(self, x_target, A_original=None, return_debug=False):
        if self.mode != "v1":
            raise NotImplementedError("CRCT currently implements only mode='v1'")
        x4 = normalize_input_x(x_target)
        B, N, T, C = x4.shape
        assert C >= 1, "CRCT x_target channel C must be >=1, got {}".format(tuple(x4.shape))
        pairs, candidate_weight = self._build_pairs(x4)
        pairs = pairs.to(x4.device)
        candidate_weight = candidate_weight.to(x4.device, dtype=x4.dtype)
        assert pairs.dim() == 2 and pairs.shape[1] == 2, "CRCT pairs must be [E,2], got {}".format(tuple(pairs.shape))

        z = self.node_encoder(x4)
        h = self.pair_encoder(z, pairs)
        logits, attr, relation_scores, known_edge_weight = self.known_heads(h)
        knownness = self.knownness_head(h, logits, attr)
        unknown_edge_weight = self.unknown_branch(h) if self.use_unknown else torch.zeros_like(known_edge_weight)
        if self.unknown_floor > 0:
            unknown_edge_weight = unknown_edge_weight.clamp_min(self.unknown_floor)
        edge_weight = knownness * known_edge_weight + (1.0 - knownness) * unknown_edge_weight

        A_crct = torch.zeros((B, N, N), device=x4.device, dtype=x4.dtype)
        A_crct[:, pairs[:, 0], pairs[:, 1]] = edge_weight
        if self.row_softmax:
            A_crct = _masked_row_softmax(A_crct)
        if self.sparse_topk > 0:
            A_crct = _row_topk(A_crct, self.sparse_topk)
        A_final = self._fuse(A_original, A_crct)

        zero = torch.zeros((), device=x4.device, dtype=x4.dtype)
        aux_loss_dict = {
            "crct_sparse_loss": sparse_loss(A_crct),
            "crct_sharp_loss": entropy_loss(attr),
            "crct_balance_loss": balance_loss(attr),
            "crct_consistency_loss": zero,
            "crct_relation_kd_loss": zero,
            "crct_unknown_reg_loss": zero,
        }
        debug_dict = self._debug_dict(
            pairs, candidate_weight, attr, knownness, known_edge_weight,
            unknown_edge_weight, edge_weight, A_original, A_crct, A_final
        ) if return_debug else {}
        return A_final, aux_loss_dict, debug_dict

    def _build_pairs(self, x4):
        B, N, T, C = x4.shape
        if self.candidate_method == "dense":
            src = torch.arange(N, device=x4.device).unsqueeze(1).expand(N, N)
            dst = torch.arange(N, device=x4.device).unsqueeze(0).expand(N, N)
            pairs = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=1).long()
            if not self.include_self_loop:
                pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            weights = torch.ones((pairs.shape[0],), device=x4.device, dtype=x4.dtype)
            return pairs, weights
        return build_candidate_edges(
            x4,
            topk=self.candidate_topk,
            method=self.candidate_method,
            include_self_loop=self.include_self_loop,
        )

    def _fuse(self, A_original, A_crct):
        if A_original is None:
            return A_crct
        A_original = A_original.to(device=A_crct.device, dtype=A_crct.dtype)
        assert A_original.dim() in [2, 3], "CRCT A_original must be [N,N] or [B,N,N], got {}".format(tuple(A_original.shape))
        if self.rho == 0.0:
            return A_original
        A_use = A_crct
        if A_original.dim() == 2:
            assert A_crct.shape[0] == 1, "CRCT cannot fuse [N,N] A_original with batched A_crct {}".format(tuple(A_crct.shape))
            A_use = A_crct.squeeze(0)
        else:
            assert A_original.shape[0] == A_crct.shape[0], "CRCT batch mismatch: A_original {}, A_crct {}".format(tuple(A_original.shape), tuple(A_crct.shape))
        assert A_original.shape[-2:] == A_use.shape[-2:], "CRCT adjacency shape mismatch: A_original {}, A_crct {}".format(tuple(A_original.shape), tuple(A_use.shape))
        return (1.0 - self.rho) * A_original + self.rho * A_use

    def _debug_dict(self, pairs, candidate_weight, attr, knownness, known_edge_weight,
                    unknown_edge_weight, edge_weight, A_original, A_crct, A_final):
        attr_mean = attr.detach().mean(dim=0)
        k = min(3, attr_mean.shape[-1])
        attr_vals, attr_ids = torch.topk(attr_mean, k=k, dim=-1)
        return {
            "pairs": pairs.detach(),
            "candidate_weight": candidate_weight.detach(),
            "A_original_stats": summarize_adj(A_original, "A_original"),
            "A_crct_stats": summarize_adj(A_crct, "A_crct"),
            "A_final_stats": summarize_adj(A_final, "A_final"),
            "knownness_stats": _summarize_vector(knownness, "knownness"),
            "relation_usage": attr.detach().mean(dim=(0, 1)),
            "attr_topk_ids": attr_ids.detach(),
            "attr_topk_values": attr_vals.detach(),
            "knownness": knownness.detach().mean(dim=0),
            "edge_known_weight": known_edge_weight.detach().mean(dim=0),
            "edge_unknown_weight": unknown_edge_weight.detach().mean(dim=0),
            "edge_final_weight": edge_weight.detach().mean(dim=0),
            "A_original_weight": _gather_edge_values(A_original, pairs),
            "A_crct_weight": _gather_edge_values(A_crct, pairs),
            "A_final_weight": _gather_edge_values(A_final, pairs),
            "target_candidate_count": int(pairs.shape[0]),
        }


def _masked_row_softmax(A):
    mask = A.abs() > 0
    logits = A.masked_fill(~mask, -1e9)
    out = F.softmax(logits, dim=-1) * mask.float()
    denom = out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out / denom


def _row_topk(A, topk):
    if topk <= 0 or topk >= A.shape[-1]:
        return A
    vals, idx = torch.topk(A, k=int(topk), dim=-1)
    out = torch.zeros_like(A)
    out.scatter_(-1, idx, vals)
    return out


def _gather_edge_values(A, pairs):
    if A is None:
        return torch.zeros((pairs.shape[0],), device=pairs.device, dtype=torch.float32)
    if A.dim() == 3:
        vals = A[:, pairs[:, 0], pairs[:, 1]].mean(dim=0)
    else:
        vals = A[pairs[:, 0], pairs[:, 1]]
    return vals.detach()


def _summarize_vector(x, name):
    x = torch.nan_to_num(x.detach().float())
    return {
        "name": name,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "sparsity": float((x.abs() <= 1e-12).float().mean().item()),
    }


def _get_arg(args, name, default):
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)
