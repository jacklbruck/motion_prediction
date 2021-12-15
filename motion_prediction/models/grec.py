import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from fairmotion.data.amass_dip import SMPL_PARENTS, SMPL_NR_JOINTS
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SAGEConv
from functools import cached_property
from torch.nn import Parameter


class GraphTransform(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=24, device="cpu"):
        super(GraphTransform, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Expand to hidden dim.
        self.l = nn.Linear(input_dim, hidden_dim)

    @cached_property
    def _edge_index(self):
        return torch.LongTensor(
            [(v, u) for u, v in enumerate(SMPL_PARENTS) if v >= 0]
        ).T

    def _get_edge_index(self, N, J):
        # Load base batch indices.
        edge_index = self._edge_index
        _, E = edge_index.size()

        # Repeat and offset by batch.
        batch_offset = torch.arange(N).unsqueeze(-1).repeat(1, E).flatten().mul(J)
        edge_index = edge_index.repeat(1, N) + batch_offset

        # Modify graph.
        edge_index = to_undirected(edge_index)
        edge_index, _ = add_self_loops(edge_index)

        return edge_index

    def _batchify(self, x):
        # Reshape for graph.
        N, _ = x.size()
        J, A = SMPL_NR_JOINTS, self.input_dim

        # Reshape for temporal batching.
        x = x.reshape(N * J, A)

        # Calculate batches.
        edge_index = self._get_edge_index(N, J).to(self.device)

        return x, edge_index

    def forward(self, x):
        x, edge_index = self._batchify(x)

        return F.relu(self.l(x)), edge_index


class GraphRecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, device="cpu"):
        super(GraphRecurrentNeuralNetwork, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self._conv_base = SAGEConv

        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_x_i = self._conv_base(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
        )

        self.conv_h_i = self._conv_base(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.output_dim))
        self.b_i = Parameter(torch.Tensor(1, self.output_dim))

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_x_f = self._conv_base(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
        )

        self.conv_h_f = self._conv_base(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.output_dim))
        self.b_f = Parameter(torch.Tensor(1, self.output_dim))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_x_c = self._conv_base(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
        )

        self.conv_h_c = self._conv_base(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
        )

        self.b_c = Parameter(torch.Tensor(1, self.output_dim))

    def _create_output_gate_parameters_and_layers(self):
        self.conv_x_o = self._conv_base(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
        )

        self.conv_h_o = self._conv_base(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.output_dim))
        self.b_o = Parameter(torch.Tensor(1, self.output_dim))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.output_dim).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.output_dim).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, H, C):
        I = self.conv_x_i(X, edge_index)
        I = I + self.conv_h_i(H, edge_index)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, H, C):
        F = self.conv_x_f(X, edge_index)
        F = F + self.conv_h_f(H, edge_index)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, H, C, I, F):
        T = self.conv_x_c(X, edge_index)
        T = T + self.conv_h_c(H, edge_index)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, H, C):
        O = self.conv_x_o(X, edge_index)
        O = O + self.conv_h_o(H, edge_index)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        h: torch.FloatTensor = None,
        c: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        # Initialize hidden state if not given.
        H = self._set_hidden_state(X, h)
        C = self._set_cell_state(X, c)

        # Calculate gates.
        I = self._calculate_input_gate(X, edge_index, H, C)
        F = self._calculate_forget_gate(X, edge_index, H, C)
        C = self._calculate_cell_state(X, edge_index, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, H, C)
        H = self._calculate_hidden_state(O, C)

        return O, (H, C)


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
            input_dim=hidden_dim, output_dim=hidden_dim, device=device
        )

        # Output layer.
        self.dec = nn.Linear(hidden_dim * SMPL_NR_JOINTS, input_dim * SMPL_NR_JOINTS)

    def init_weights(self):
        pass

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        # Pass inputs through recurrent network.
        for t in range(src.size(1) - 12, src.size(1)):
            if t == src.size(1) - 12 or np.random.random() < 1/6:
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
