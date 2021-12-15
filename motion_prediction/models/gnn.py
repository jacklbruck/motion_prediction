import torch.nn.functional as F
import torch.nn as nn
import torch

from fairmotion.data.amass_dip import SMPL_PARENTS, SMPL_NR_JOINTS
from torch_geometric.nn import GATConv


class Encoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=24, num_layers=1, device="cpu"):
        super(Encoder, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.edge_index = self.get_edge_index()

        # Preprocessing linear layer.
        self.pre = nn.Linear(input_dim, hidden_dim)

        # Graph layer.
        self.gc = GATConv(in_channels=hidden_dim, out_channels=hidden_dim // 3, heads=3)

    def get_edge_index(self):
        return torch.LongTensor(
            [(v, u) for u, v in enumerate(SMPL_PARENTS) if v >= 0]
        ).T.to(self.device)

    def _to_graph_seq(self, X):
        N, L, D = X.size()

        return X.reshape(N, L, SMPL_NR_JOINTS, D // SMPL_NR_JOINTS)

    def from_graph_seq(self, X, dims):
        N, L, J, _ = dims

        return X.reshape(N, L, J * self.hidden_dim)

    def batchify(self, X):
        # Reshape for graph.
        X = self._to_graph_seq(X)

        # Get dimension.
        N, L, J, A = X.size()
        dims = (N, L, J, A)

        X = X.reshape(N * L * J, A)
        edge_index = self.edge_index.repeat(1, N * L) + (
            torch.arange(N * L) * J
        ).unsqueeze(-1).repeat(1, len(self.edge_index.T)).flatten().to(self.device)
        batch = (
            torch.arange(N).repeat(L * J).reshape(L * J, -1).T.flatten().to(self.device)
        )

        return X, edge_index, batch, dims

    def forward(self, src):
        # Process for graph input.
        X, edge_index, _, dims = self.batchify(src)
        (N, L, J, A) = dims

        # Pass through graph neural netowrk.
        X = F.relu(self.pre(X))
        X = F.relu(self.gc(X, edge_index))
        X = X.reshape(N, L, J * self.hidden_dim)

        return X


class Model(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=24, num_layers=1, device="cpu"):
        super(Model, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Build layers.
        self.enc = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device
        )
        self.rec = nn.LSTM(
            input_size=hidden_dim * SMPL_NR_JOINTS,
            hidden_size=hidden_dim * SMPL_NR_JOINTS,
            num_layers=1,
            batch_first=True,
        )
        self.dec = nn.Linear(hidden_dim * SMPL_NR_JOINTS, input_dim * SMPL_NR_JOINTS)

    def init_weights(self):
        pass

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        # Encode input sequences.
        out = self.enc(src)
        tgt = self.enc(tgt)
        _, (h, c) = self.rec(out)

        max_len = max_len if max_len is not None else tgt.size(1)

        out = tgt[:, 0].unsqueeze(1)
        outs = torch.zeros(tgt.size(0), max_len, self.hidden_dim * SMPL_NR_JOINTS).to(
            self.device
        )
        for t in range(max_len):
            out, (h, c) = self.rec(out, (h, c))

            outs[:, t] = out.squeeze(1)

        return self.dec(outs)
