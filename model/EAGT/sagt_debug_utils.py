import csv
from pathlib import Path


def dump_sagt_csv(debug_dict, source_structure_cache, path, top_edges=30):
    """Dump SAGT target-edge source attribution CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = debug_dict["target_pairs"].detach().cpu()
    city_names = source_structure_cache.get_city_names() if source_structure_cache is not None else []
    exemplars = source_structure_cache.get_exemplars() if source_structure_cache is not None else {}
    rho = debug_dict["source_city_weight"].detach().cpu()
    top_city_idx = debug_dict["top_source_city_idx"].detach().cpu()
    rows = []
    for e in range(min(int(top_edges), pairs.shape[0])):
        city_idx = int(top_city_idx[e].item()) if top_city_idx.numel() > 0 else -1
        city = city_names[city_idx] if 0 <= city_idx < len(city_names) else ""
        city_weight = float(rho[city_idx].item()) if 0 <= city_idx < rho.numel() else 0.0
        city_exemplars = exemplars.get(city, [])
        if not city_exemplars:
            city_exemplars = [{}]
        for rank, meta in enumerate(city_exemplars[:3]):
            rows.append({
                "target_i": int(pairs[e, 0].item()),
                "target_j": int(pairs[e, 1].item()),
                "final_score": _edge(debug_dict, "A_final_weight", e, default_key="score_lowrank"),
                "lowrank_score": _edge(debug_dict, "score_lowrank", e),
                "src_role_score": _edge(debug_dict, "score_src_role", e),
                "eagt_score": _edge(debug_dict, "score_eagt", e),
                "res_score": _edge(debug_dict, "score_res", e),
                "source_ratio": _edge(debug_dict, "source_ratio", e),
                "residual_ratio": _edge(debug_dict, "residual_ratio", e),
                "top_source_city": city,
                "top_source_city_weight": city_weight,
                "rank": rank,
                "source_city": meta.get("source_city", city),
                "source_u": meta.get("source_u", -1),
                "source_v": meta.get("source_v", -1),
                "source_role_score": meta.get("source_role_score", 0.0),
                "source_adj_weight": meta.get("source_adj_weight", 0.0),
            })
    fieldnames = [
        "target_i", "target_j", "final_score", "lowrank_score", "src_role_score",
        "eagt_score", "res_score", "source_ratio", "residual_ratio",
        "top_source_city", "top_source_city_weight", "rank", "source_city",
        "source_u", "source_v", "source_role_score", "source_adj_weight",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _edge(debug_dict, key, idx, default_key=None):
    value = debug_dict.get(key, None)
    if value is None and default_key is not None:
        value = debug_dict.get(default_key, None)
    if value is None:
        return 0.0
    value = value.detach().cpu()
    if value.dim() == 0:
        return float(value.item())
    return float(value[idx].item())
