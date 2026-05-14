import csv
from pathlib import Path

import torch


def summarize_adj(A, name="A"):
    """Return min/max/mean/sparsity stats for adjacency [N,N] or [B,N,N]."""
    if A is None:
        return {"name": name, "min": 0.0, "max": 0.0, "mean": 0.0, "sparsity": 1.0}
    A = A.detach()
    finite = torch.nan_to_num(A.float())
    return {
        "name": name,
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "sparsity": float((finite.abs() <= 1e-12).float().mean().item()),
    }


def print_eagt_shapes(prefix="[EAGT]", **kwargs):
    """Print tensor shapes with the unified EAGT prefix."""
    parts = []
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            parts.append("{}={}".format(key, tuple(value.shape)))
        else:
            parts.append("{}={}".format(key, value))
    print("{} {}".format(prefix, ", ".join(parts)))


def dump_evidence_csv(debug_dict, source_metadata, path, top_edges=20):
    """
    Dump target-edge evidence attribution.

    CSV fields:
    target_i,target_j,target_edge_weight,rank,source_city,source_u,source_v,
    source_importance,similarity,alpha,is_physical_edge
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = debug_dict["target_pairs"].detach().cpu()
    weights = debug_dict["target_edge_weight"].detach().cpu()
    idx = debug_dict["topk_source_idx"].detach().cpu()
    score = debug_dict["topk_score"].detach().cpu()
    alpha = debug_dict["topk_alpha"].detach().cpu()
    assert pairs.dim() == 2 and pairs.shape[1] == 2, "target_pairs must be [E,2]"
    rows = []
    for e in range(min(int(top_edges), pairs.shape[0])):
        ti, tj = int(pairs[e, 0].item()), int(pairs[e, 1].item())
        for rank in range(idx.shape[1]):
            meta = source_metadata[int(idx[e, rank].item())]
            rows.append({
                "target_i": ti,
                "target_j": tj,
                "target_edge_weight": float(weights[e].item()),
                "rank": rank,
                "source_city": meta.get("source_city", ""),
                "source_u": meta.get("source_u", -1),
                "source_v": meta.get("source_v", -1),
                "source_importance": meta.get("source_importance", 0.0),
                "similarity": float(score[e, rank].item()),
                "alpha": float(alpha[e, rank].item()),
                "is_physical_edge": meta.get("is_physical_edge", -1),
            })
    fieldnames = [
        "target_i", "target_j", "target_edge_weight", "rank", "source_city",
        "source_u", "source_v", "source_importance", "similarity", "alpha",
        "is_physical_edge",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

