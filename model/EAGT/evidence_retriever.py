import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceRetriever(nn.Module):
    """
    Chunked cosine retrieval from source edge evidence.

    Input:
        target_edge_feats: FloatTensor [E_t, F].
        source_edge_feats: FloatTensor [E_s, F].
        source_importance: optional FloatTensor [E_s].
    Output:
        topk_idx: LongTensor [E_t, K].
        topk_score: FloatTensor [E_t, K].
        topk_alpha: FloatTensor [E_t, K].
    """
    def __init__(self, w_importance=0.1, w_grad=0.0):
        super(EvidenceRetriever, self).__init__()
        self.w_importance = float(w_importance)
        self.w_grad = float(w_grad)

    def forward(self, target_edge_feats, source_edge_feats, source_importance=None,
                topk=8, chunk_size=4096, random=False):
        assert target_edge_feats.dim() == 2, "target_edge_feats must be [E_t,F]"
        assert source_edge_feats.dim() == 2, "source_edge_feats must be [E_s,F]"
        assert target_edge_feats.shape[1] == source_edge_feats.shape[1], (
            "feature dim mismatch: target {}, source {}".format(target_edge_feats.shape, source_edge_feats.shape)
        )
        E_t, E_s = target_edge_feats.shape[0], source_edge_feats.shape[0]
        assert E_s > 0, "EAGT source evidence cache is empty"
        k = min(max(1, int(topk)), E_s)
        device = target_edge_feats.device
        source_edge_feats = source_edge_feats.to(device)
        if source_importance is None:
            source_importance = torch.zeros(E_s, device=device, dtype=target_edge_feats.dtype)
        else:
            source_importance = source_importance.to(device=device, dtype=target_edge_feats.dtype)
        assert source_importance.shape[0] == E_s, "source_importance must be [E_s]"

        if random:
            idx = torch.randint(0, E_s, (E_t, k), device=device)
            score = torch.rand((E_t, k), device=device, dtype=target_edge_feats.dtype)
            alpha = F.softmax(score, dim=-1)
            return idx.long(), score, alpha

        source_norm = F.normalize(source_edge_feats, p=2, dim=1)
        target_norm = F.normalize(target_edge_feats, p=2, dim=1)
        idx_chunks, score_chunks = [], []
        chunk_size = max(1, int(chunk_size))
        for start in range(0, E_t, chunk_size):
            end = min(E_t, start + chunk_size)
            sim = torch.matmul(target_norm[start:end], source_norm.t())
            score = sim + self.w_importance * source_importance.view(1, -1)
            vals, idx = torch.topk(score, k=k, dim=1)
            idx_chunks.append(idx.long())
            score_chunks.append(vals)
        topk_idx = torch.cat(idx_chunks, dim=0)
        topk_score = torch.cat(score_chunks, dim=0)
        topk_alpha = F.softmax(topk_score, dim=-1)
        assert topk_idx.shape == (E_t, k), "topk_idx must be [E_t,K]"
        return topk_idx, topk_score, topk_alpha

