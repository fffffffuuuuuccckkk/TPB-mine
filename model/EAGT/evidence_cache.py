from pathlib import Path

import torch

from .edge_features import build_candidate_edges, edge_feature_from_pairs, compute_corr_matrix


class SourceEvidenceCache(object):
    """
    Stores source-city edge evidence.

    features: FloatTensor [E_s, F].
    metadata: list[dict] length E_s, one entry per concrete source edge.
    """
    def __init__(self, cache_dir="./save/eagt_cache", device="cpu"):
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device)
        self.features = torch.empty((0, 8), dtype=torch.float32, device=self.device)
        self.metadata = []

    def build_from_source_data(self, source_data_dict, source_adj_dict=None, args=None):
        """
        Build edge evidence from source city histories.

        Args:
            source_data_dict: {city: x}, each x [B,N,T,C], [N,T,C], or tensor-compatible.
            source_adj_dict: optional {city: adjacency}, each [N,N].
            args: object/dict with eagt_source_topk_per_node and eagt_max_source_edges.
        """
        assert isinstance(source_data_dict, dict) and len(source_data_dict) > 0, "EAGT source_data_dict must be non-empty"
        source_adj_dict = source_adj_dict or {}
        topk = _get_arg(args, "eagt_source_topk_per_node", 20)
        max_edges = _get_arg(args, "eagt_max_source_edges", 200000)
        method = _get_arg(args, "eagt_candidate_method", "corr")
        include_self = bool(_get_arg(args, "eagt_include_self_loop", 0))

        feature_chunks, metadata = [], []
        seen = set()
        for city, x in source_data_dict.items():
            x = torch.as_tensor(x, dtype=torch.float32)
            pairs, _ = build_candidate_edges(x, topk=topk, method=method, include_self_loop=include_self)
            adj = source_adj_dict.get(city, None)
            if adj is not None:
                adj_t = torch.as_tensor(adj, dtype=torch.float32)
                phys_pairs = torch.nonzero(adj_t > 0, as_tuple=False).long()
                if not include_self and phys_pairs.numel() > 0:
                    phys_pairs = phys_pairs[phys_pairs[:, 0] != phys_pairs[:, 1]]
                if phys_pairs.numel() > 0:
                    pairs = torch.cat([pairs.cpu(), phys_pairs.cpu()], dim=0)
            pairs = _unique_pairs(pairs)
            feats = edge_feature_from_pairs(x, pairs, method="corr_lagcorr").cpu()
            corr = compute_corr_matrix(x).cpu()
            adj_t = torch.as_tensor(adj, dtype=torch.float32).cpu() if adj is not None else None
            for row, (u, v) in enumerate(pairs.cpu().tolist()):
                key = (str(city), int(u), int(v))
                if key in seen:
                    continue
                seen.add(key)
                source_weight = float(corr[u, v].item())
                is_physical = -1
                if adj_t is not None:
                    is_physical = int(adj_t[u, v].item() > 0)
                    if is_physical:
                        source_weight = float(adj_t[u, v].item())
                source_importance = abs(source_weight)
                feature_chunks.append(feats[row:row + 1])
                metadata.append({
                    "source_city": str(city),
                    "source_u": int(u),
                    "source_v": int(v),
                    "is_physical_edge": is_physical,
                    "source_weight": source_weight,
                    "source_importance": source_importance,
                })

        if feature_chunks:
            features = torch.cat(feature_chunks, dim=0).float()
        else:
            features = torch.empty((0, 8), dtype=torch.float32)
        assert features.dim() == 2, "EAGT source features must be [E_s,F]"
        if features.shape[0] > int(max_edges):
            importance = torch.tensor([m["source_importance"] for m in metadata], dtype=torch.float32)
            _, idx = torch.topk(importance, k=int(max_edges))
            idx_sorted = idx.sort()[0]
            features = features[idx_sorted]
            metadata = [metadata[int(i)] for i in idx_sorted.tolist()]
        self.features = features.to(self.device)
        self.metadata = metadata
        return self

    def save(self, path):
        """Save cache to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": self.features.cpu(), "metadata": self.metadata}, path)
        return path

    def load(self, path):
        """Load cache from a .pt file."""
        data = torch.load(path, map_location="cpu")
        self.features = data["features"].float().to(self.device)
        self.metadata = data["metadata"]
        assert self.features.shape[0] == len(self.metadata), "EAGT cache features/metadata length mismatch"
        return self

    def to(self, device):
        """Move feature tensor to device."""
        self.device = torch.device(device)
        self.features = self.features.to(self.device)
        return self

    def get_features(self):
        """Return source evidence features [E_s,F]."""
        return self.features

    def get_metadata(self):
        """Return source evidence metadata list."""
        return self.metadata


def _get_arg(args, name, default):
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _unique_pairs(pairs):
    pairs = pairs.long().cpu()
    if pairs.numel() == 0:
        return pairs.reshape(0, 2)
    seen, out = set(), []
    for u, v in pairs.tolist():
        key = (int(u), int(v))
        if key not in seen:
            seen.add(key)
            out.append([int(u), int(v)])
    return torch.tensor(out, dtype=torch.long)
