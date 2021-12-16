import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from fairmotion.data.amass_dip import SMPL_PARENTS, SMPL_NR_JOINTS
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch, Data
from functools import cached_property
from torch_sparse import matmul
from torch.nn import Parameter


class GraphTransform(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=24, device="cpu"):
        super(GraphTransform, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Layer to expand input to hidden dim.
        self.l = nn.Linear(input_dim, hidden_dim)

    def _get_edge_index(self, N, J):
        # Load base batch indices.
        edge_index = torch.LongTensor(
            [(v, u) for u, v in enumerate(SMPL_PARENTS) if v >= 0]
        ).T
        _, E = edge_index.size()

        # Repeat and offset by batch.
        batch_offset = torch.arange(N).unsqueeze(-1).repeat(1, E).flatten().mul(J)
        edge_index = edge_index.repeat(1, N) + batch_offset

        # Modify graph.
        edge_index = to_undirected(edge_index)
        edge_index, _ = add_self_loops(edge_index)

        return edge_index

    def _batchify(self, x):
        return x.reshape(-1, self.input_dim)

    def forward(self, x):
        edge_index = self._get_edge_index(x.size(0), SMPL_NR_JOINTS).to(self.device)
        x = self._batchify(x)

        return F.relu(self.l(x)), edge_index


class GraphConvolution(MessagePassing):
    def __init__(self, input_dim=3, hidden_dim=24):
        super(GraphConvolution, self).__init__(aggr="mean")

        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Source and target nodes.
        self.l = nn.Linear(input_dim, hidden_dim, bias=True)
        self.r = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=(x, x), size=None)
        out = self.l(out) + self.r(x)

        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)


class GraphRecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, device="cpu"):
        super(GraphRecurrentNeuralNetwork, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Initialize layers.
        self.l = nn.ModuleDict()
        for g in ["i", "f", "c", "o"]:
            # Initialize gate.
            self.l[g] = nn.ModuleDict()

            # Convolutional layers.
            self.l[g]["x"] = GraphConvolution(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
            )
            self.l[g]["h"] = GraphConvolution(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
            )

        # Initialize weight parameters.
        self.w = nn.ParameterDict()
        for g in ["i", "f", "c", "o"]:
            if g == "c":
                continue

            self.w[g] = Parameter(torch.Tensor(1, self.hidden_dim))
            glorot(self.w[g])

        # Initialize bias parameters.
        self.b = nn.ParameterDict()
        for g in ["i", "f", "c", "o"]:
            self.b[g] = Parameter(torch.Tensor(1, self.hidden_dim))
            zeros(self.b[g])

    def _i(self, X, edge_index, h, c):
        return torch.sigmoid(
            self.l["i"]["x"](X, edge_index)
            + self.l["i"]["h"](h, edge_index)
            + self.w["i"] * c
            + self.b["i"]
        )

    def _f(self, X, edge_index, h, c):
        return torch.sigmoid(
            self.l["f"]["x"](X, edge_index)
            + self.l["f"]["h"](h, edge_index)
            + self.w["f"] * c
            + self.b["f"]
        )

    def _c(self, X, edge_index, h, c, i, f):
        return f * c + i * torch.tanh(
            self.l["c"]["x"](X, edge_index)
            + self.l["c"]["h"](h, edge_index)
            + self.b["c"]
        )

    def _o(self, X, edge_index, h, c):
        return torch.sigmoid(
            self.l["o"]["x"](X, edge_index)
            + self.l["o"]["h"](h, edge_index)
            + self.w["o"] * c
            + self.b["o"]
        )

    def _h(self, O, C):
        return O * torch.tanh(C)

    def forward(self, X, edge_index, h=None, c=None):
        # Initialize hidden state if not given.
        h = torch.zeros(X.size(0), self.hidden_dim).to(X.device) if h is None else h
        c = torch.zeros(X.size(0), self.hidden_dim).to(X.device) if c is None else c

        # Calculate gates.
        i = self._i(X, edge_index, h, c)
        f = self._f(X, edge_index, h, c)
        c = self._c(X, edge_index, h, c, i, f)
        o = self._o(X, edge_index, h, c)
        h = self._h(o, c)

        return o, (h, c)


class Model(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=24, num_layers=1, device="cpu"):
        super(Model, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Preprocessing layer.
        self.enc = GraphTransform(
            input_dim=input_dim, hidden_dim=hidden_dim, device=device
        )

        # Recurrent layer.
        self.rec = GraphRecurrentNeuralNetwork(
            input_dim=hidden_dim, hidden_dim=hidden_dim, device=device
        )

        # Output layer.
        self.dec = nn.Linear(hidden_dim * SMPL_NR_JOINTS, input_dim * SMPL_NR_JOINTS)

    def init_weights(self):
        pass

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        # Pass inputs through recurrent network.
        for t in range(src.size(1) - 24, src.size(1)):
            if t == src.size(1) - 24 or np.random.random() < 1 / 6:
                inp, edge_index = self.enc(src[:, t])

                out, (h, c) = self.rec(inp, edge_index)
            else:
                out, (h, c) = self.rec(out, edge_index, h=h, c=c)

        # Initialize output parameters.
        outs = torch.zeros(
            tgt.size(0),
            max_len if max_len is not None else tgt.size(1),
            self.hidden_dim * SMPL_NR_JOINTS,
        ).to(self.device)

        for t in range(outs.size(1)):
            if not t:
                inp, edge_index = self.enc(tgt[:, 0])
            else:
                inp = out

            out, (h, c) = self.rec(inp, edge_index, h=h, c=c)
            outs[:, t] = out.reshape(src.size(0), SMPL_NR_JOINTS * self.hidden_dim)

        return self.dec(outs)
