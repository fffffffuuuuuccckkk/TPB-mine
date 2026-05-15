import csv
from pathlib import Path

import torch


def summarize_adj(A, name="A"):
    """Return min/max/mean/sparsity stats for adjacency [N,N] or [B,N,N]."""
    if A is None:
        return {"name": name, "min": 0.0, "max": 0.0, "mean": 0.0, "sparsity": 1.0}
    finite = torch.nan_to_num(A.detach().float())
    return {
        "name": name,
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "sparsity": float((finite.abs() <= 1e-12).float().mean().item()),
    }


def print_crct_shapes(prefix="[CRCT]", **kwargs):
    """Print tensor shapes with the unified CRCT prefix."""
    parts = []
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            parts.append("{}={}".format(key, tuple(value.shape)))
        else:
            parts.append("{}={}".format(key, value))
    print("{} {}".format(prefix, ", ".join(parts)))


def dump_crct_csv(debug_dict, path, top_edges=20):
    """
    Dump CRCT relation attribution per target edge.

    CSV fields:
    target_i,target_j,A_original_weight,A_crct_weight,A_final_weight,knownness,
    unknown_weight,top_relation_1,top_relation_weight_1,...,relation_usage_summary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = debug_dict["pairs"].detach().cpu()
    top_ids = debug_dict["attr_topk_ids"].detach().cpu()
    top_vals = debug_dict["attr_topk_values"].detach().cpu()
    usage = debug_dict["relation_usage"].detach().cpu()
    usage_summary = ";".join(["{}:{:.6f}".format(i, float(v)) for i, v in enumerate(usage.tolist())])
    rows = []
    for e in range(min(int(top_edges), pairs.shape[0])):
        row = {
            "target_i": int(pairs[e, 0].item()),
            "target_j": int(pairs[e, 1].item()),
            "A_original_weight": _edge_value(debug_dict, "A_original_weight", e),
            "A_crct_weight": _edge_value(debug_dict, "A_crct_weight", e),
            "A_final_weight": _edge_value(debug_dict, "A_final_weight", e),
            "knownness": _edge_value(debug_dict, "knownness", e),
            "unknown_weight": _edge_value(debug_dict, "edge_unknown_weight", e),
            "top_relation_1": int(top_ids[e, 0].item()) if top_ids.shape[1] > 0 else -1,
            "top_relation_weight_1": float(top_vals[e, 0].item()) if top_vals.shape[1] > 0 else 0.0,
            "top_relation_2": int(top_ids[e, 1].item()) if top_ids.shape[1] > 1 else -1,
            "top_relation_weight_2": float(top_vals[e, 1].item()) if top_vals.shape[1] > 1 else 0.0,
            "top_relation_3": int(top_ids[e, 2].item()) if top_ids.shape[1] > 2 else -1,
            "top_relation_weight_3": float(top_vals[e, 2].item()) if top_vals.shape[1] > 2 else 0.0,
            "relation_usage_summary": usage_summary,
        }
        rows.append(row)
    fieldnames = [
        "target_i", "target_j", "A_original_weight", "A_crct_weight",
        "A_final_weight", "knownness", "unknown_weight",
        "top_relation_1", "top_relation_weight_1",
        "top_relation_2", "top_relation_weight_2",
        "top_relation_3", "top_relation_weight_3",
        "relation_usage_summary",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def dump_relation_usage(debug_dict, path):
    """Dump mean relation head usage [K] to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = debug_dict["relation_usage"].detach().cpu()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relation_id", "mean_usage"])
        writer.writeheader()
        for i, value in enumerate(usage.tolist()):
            writer.writerow({"relation_id": i, "mean_usage": float(value)})
    return path


def _edge_value(debug_dict, key, idx):
    value = debug_dict.get(key, None)
    if value is None:
        return 0.0
    value = value.detach().cpu()
    return float(value[idx].item())
