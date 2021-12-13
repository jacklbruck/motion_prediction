import torch.nn.functional as F
import torch.nn as nn
import torch

from fairmotion.data.amass_dip import SMPL_JOINT_MAPPING, SMPL_PARENTS, SMPL_NR_JOINTS
from torch_geometric.utils import to_undirected
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv
from functools import cached_property

class GraphModelStep(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=1, n=4, device="cpu"):
        super(GraphModelStep, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.n = n

        # Graph layers.
        self.enc = GATConv(in_channels=input_dim, out_channels=hidden_dim)
        self.post = nn.Linear(in_features=hidden_dim, out_features=input_dim)

    def _to_graph_seq(self, X):
        N, L, D = X.size()

        return X.reshape(N, L, SMPL_NR_JOINTS, D // SMPL_NR_JOINTS)

    def _from_graph_seq(self, X):
        return X.reshape(2 ** self.n, -1, SMPL_NR_JOINTS, self.hidden_dim)

    @cached_property
    def _edge_index(self):
        return torch.LongTensor([(v, u) for u, v in enumerate(SMPL_PARENTS) if v >= 0])

    def _get_edge_index(self, L, J):
        # Load base batch indices.
        edge_index = self._edge_index.T
        _, E = edge_index.size()

        # Repeat and offset by batch.
        batch_offset = torch.arange(L).unsqueeze(-1).repeat(1, E).flatten().mul(J)
        edge_index = edge_index.repeat(1, L) + batch_offset

        # Modify graph.
        edge_index = to_undirected(edge_index)
        for n in range(self.n):
            time_edges = torch.stack(
                [torch.arange(J * (L - 2 ** n)), torch.arange(2 ** n * J, J * L)]
            )

            edge_index = torch.cat([edge_index, time_edges], dim=1)

        return edge_index

    def _batchify(self, X):
        # Reshape for graph.
        X = self._to_graph_seq(X)

        # Get dimension.
        N, L, J, A = X.size()

        # Collapse time dimension.
        X = X.reshape(N, L * J, A)

        # Calculate batches.
        edge_index = self._get_edge_index(L, J).to(self.device)
        data_list = [Data(X[t], edge_index) for t in range(N)]

        return Batch.from_data_list(data_list)

    def forward(self, src):
        # Batch source and tgt sequences by time.
        srcb = self._batchify(src)

        # Pass through model
        out = self.enc(srcb.x, srcb.edge_index)
        out = out.reshape(src.size(0), src.size(1), SMPL_NR_JOINTS, self.hidden_dim)

        if self.training:
            out = self.post(out[:, 2 ** self.n : -1]).reshape(
                src.size(0),
                src.size(1) - 2 ** self.n - 1,
                SMPL_NR_JOINTS * self.input_dim,
            )
        else:
            out = self.post(out[:, -1]).reshape(-1, SMPL_NR_JOINTS * self.input_dim)

        return out



class GraphModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=1, n=4, device="cpu"):
        super(GraphModel, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n = n
        self.device = device

        # Graph layers.
        self.gat = GraphModelStep(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, n=n, device=device)

    def init_weights(self):
        pass

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=None):
        # Generate sequence.
        if self.training:
            outs = self.gat(torch.cat((src, tgt), dim=1))
        else:
            # Prepare output seq.
            max_len = tgt.size(1) if max_len is None else max_len
            outs = torch.zeros(tgt.size(0), max_len, tgt.size(2)).to(self.device)

            for t in range(max_len):
                if not t:
                    seed = src[:, -2 ** self.n:]
                elif t < 2 ** self.n:
                    seed = torch.cat([src[:, -2 ** self.n + t:], outs[:, :t]], dim=1)
                else:
                    seed = outs[:, t - 2 ** self.n:t]

                # Pass through networks.
                outs[:, t, :] = self.gat(seed)

        return outs