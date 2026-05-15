import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_ops import sparsemax


class KnownRelationHeads(nn.Module):
    """
    Source-known relation heads.

    Input:
        h: Tensor [B,E,R].
    Output:
        logits: Tensor [B,E,K].
        attr: Tensor [B,E,K].
        relation_edge_scores: Tensor [B,E,K].
        known_edge_weight: Tensor [B,E].
    """
    def __init__(self, relation_dim=64, num_relations=8,
                 attribution="sparsemax", temperature=1.0):
        super(KnownRelationHeads, self).__init__()
        assert attribution in ["softmax", "sparsemax", "entmax15"], "unsupported crct_attribution {}".format(attribution)
        self.num_relations = int(num_relations)
        self.attribution = attribution
        self.temperature = max(float(temperature), 1e-6)
        self.logit_layer = nn.Linear(relation_dim, self.num_relations)
        self.score_layer = nn.Linear(relation_dim, self.num_relations)
        self._warned_entmax = False

    def forward(self, h):
        assert h.dim() == 3, "CRCT h must be [B,E,R], got {}".format(tuple(h.shape))
        logits = self.logit_layer(h)
        scaled = logits / self.temperature
        if self.attribution == "softmax":
            attr = F.softmax(scaled, dim=-1)
        else:
            if self.attribution == "entmax15" and not self._warned_entmax:
                print("[CRCT] entmax15 is not implemented in v1; fallback to sparsemax.")
                self._warned_entmax = True
            attr = sparsemax(scaled, dim=-1)
        relation_edge_scores = torch.sigmoid(self.score_layer(h))
        known_edge_weight = (attr * relation_edge_scores).sum(dim=-1)
        return logits, attr, relation_edge_scores, known_edge_weight


class KnownnessHead(nn.Module):
    """
    Estimate whether a target edge is explainable by known source relations.

    Input:
        h: Tensor [B,E,R], logits/attr: Tensor [B,E,K].
    Output:
        knownness: Tensor [B,E] in [0,1].
    """
    def __init__(self, relation_dim=64, num_relations=8, method="entropy"):
        super(KnownnessHead, self).__init__()
        assert method in ["maxlogit", "entropy", "mlp"], "unsupported crct_knownness_method {}".format(method)
        self.method = method
        self.num_relations = int(num_relations)
        if method == "mlp":
            self.model = nn.Sequential(
                nn.Linear(relation_dim + 2, relation_dim),
                nn.ReLU(),
                nn.Linear(relation_dim, 1),
            )
        else:
            self.model = None

    def forward(self, h, logits, attr):
        eps = 1e-8
        entropy = -(attr.clamp_min(eps) * attr.clamp_min(eps).log()).sum(dim=-1)
        if self.method == "maxlogit":
            return torch.sigmoid(logits.max(dim=-1).values)
        if self.method == "entropy":
            denom = math.log(max(2, self.num_relations))
            return (1.0 - entropy / denom).clamp(0.0, 1.0)
        maxlogit = logits.max(dim=-1).values.unsqueeze(-1)
        entropy_feat = entropy.unsqueeze(-1)
        knownness = torch.sigmoid(self.model(torch.cat([h, maxlogit, entropy_feat], dim=-1))).squeeze(-1)
        return knownness.clamp(0.0, 1.0)


class UnknownEdgeBranch(nn.Module):
    """
    Target-private edge branch.

    Input:
        h: Tensor [B,E,R].
    Output:
        unknown_edge_weight: Tensor [B,E] in [0,1].
    """
    def __init__(self, relation_dim=64, dropout=0.1):
        super(UnknownEdgeBranch, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(relation_dim, 1),
        )

    def forward(self, h):
        assert h.dim() == 3, "CRCT h must be [B,E,R], got {}".format(tuple(h.shape))
        return torch.sigmoid(self.model(h)).squeeze(-1)
