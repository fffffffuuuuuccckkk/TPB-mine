import torch
import torch.nn as nn

from EAGT.edge_features import normalize_input_x


class NodeTemporalEncoder(nn.Module):
    """
    Encode node historical series.

    Input:
        x: Tensor [B,N,T,C] or [N,T,C].
    Output:
        z: Tensor [B,N,D], where D=hidden_dim.
    """
    def __init__(self, encoder_type="tcn", input_channels=1, window_size=288,
                 hidden_dim=64, dropout=0.1):
        super(NodeTemporalEncoder, self).__init__()
        assert encoder_type in ["mlp", "tcn", "gru"], "unsupported crct_node_encoder {}".format(encoder_type)
        self.encoder_type = encoder_type
        self.input_channels = int(input_channels)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        if encoder_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(self.window_size * self.input_channels, hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif encoder_type == "tcn":
            self.model = nn.Sequential(
                nn.Conv1d(self.input_channels, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.model = nn.GRU(
                input_size=self.input_channels,
                hidden_size=hidden_dim,
                batch_first=True,
            )

    def forward(self, x):
        x4 = normalize_input_x(x)
        B, N, T, C = x4.shape
        assert C == self.input_channels, "CRCT expected C={}, got x shape {}".format(self.input_channels, tuple(x4.shape))
        if self.encoder_type == "mlp":
            assert T == self.window_size, "CRCT mlp expected T={}, got x shape {}".format(self.window_size, tuple(x4.shape))
            z = self.model(x4.reshape(B * N, T * C)).reshape(B, N, self.hidden_dim)
        elif self.encoder_type == "tcn":
            xt = x4.reshape(B * N, T, C).permute(0, 2, 1)
            h = self.model(xt).mean(dim=-1)
            z = h.reshape(B, N, self.hidden_dim)
        else:
            xt = x4.reshape(B * N, T, C)
            _, h = self.model(xt)
            z = h[-1].reshape(B, N, self.hidden_dim)
        assert z.shape == (B, N, self.hidden_dim), "CRCT z must be [B,N,D], got {}".format(tuple(z.shape))
        return z


class PairRelationEncoder(nn.Module):
    """
    Encode directed node pairs into relation embeddings.

    Input:
        z: Tensor [B,N,D].
        pairs: LongTensor [E,2].
    Output:
        h: Tensor [B,E,relation_dim].
    """
    def __init__(self, node_dim=64, relation_dim=64, dropout=0.1):
        super(PairRelationEncoder, self).__init__()
        self.node_dim = int(node_dim)
        self.relation_dim = int(relation_dim)
        self.model = nn.Sequential(
            nn.Linear(self.node_dim * 4, self.relation_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.relation_dim, self.relation_dim),
        )

    def forward(self, z, pairs, manual_feats=None):
        assert z.dim() == 3, "CRCT z must be [B,N,D], got {}".format(tuple(z.shape))
        pairs = pairs.long().to(z.device)
        assert pairs.dim() == 2 and pairs.shape[1] == 2, "CRCT pairs must be [E,2], got {}".format(tuple(pairs.shape))
        B, N, D = z.shape
        assert D == self.node_dim, "CRCT node dim mismatch: expected {}, got {}".format(self.node_dim, D)
        if pairs.numel() > 0:
            assert int(pairs.max().item()) < N and int(pairs.min().item()) >= 0, "CRCT pair index out of range for N={}".format(N)
        zi = z[:, pairs[:, 0], :]
        zj = z[:, pairs[:, 1], :]
        pair = torch.cat([zi, zj, zi - zj, zi * zj], dim=-1)
        h = self.model(pair)
        assert h.shape == (B, pairs.shape[0], self.relation_dim), "CRCT h must be [B,E,R], got {}".format(tuple(h.shape))
        return h
