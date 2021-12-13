import torch.nn as nn
import numpy as np
import torch

from fairmotion.data.amass_dip import SMPL_JOINT_MAPPING, SMPL_PARENTS, SMPL_NR_JOINTS
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Batch, Data
from torch_geometric.nn import ChebConv
from functools import cached_property
from torch.nn import Parameter


class GConvLSTM(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

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
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C):
        I = self.conv_x_i(X, edge_index, edge_weight)
        I = I + self.conv_h_i(H, edge_index, edge_weight)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C):
        F = self.conv_x_f(X, edge_index, edge_weight)
        F = F + self.conv_h_f(H, edge_index, edge_weight)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F):
        T = self.conv_x_c(X, edge_index, edge_weight)
        T = T + self.conv_h_c(H, edge_index, edge_weight)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C):
        O = self.conv_x_o(X, edge_index, edge_weight)
        O = O + self.conv_h_o(H, edge_index, edge_weight)
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
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C)
        H = self._calculate_hidden_state(O, C)
        return O, (H, C)

class GraphModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=1, device="cpu"):
        super(GraphModel, self).__init__()
        # Save parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Recurrent layer.
        self.enc = GConvLSTM(in_channels=input_dim, out_channels=hidden_dim, K=3)
        self.dec = GConvLSTM(in_channels=input_dim, out_channels=hidden_dim, K=3)
        self.post = nn.Linear(in_features=hidden_dim, out_features=input_dim)

    def init_weights(self):
        pass

    @cached_property
    def _edge_index_base(self):
        return torch.LongTensor([(v, u) for u, v in enumerate(SMPL_PARENTS) if v >= 0])

    def _get_edge_index(self, N, J):
        # Load base batch indices.
        edge_index = self._edge_index_base.T
        _, E = edge_index.size()

        # Repeat and offset by batch.
        batch_offset = torch.arange(N).unsqueeze(-1).repeat(1, E).flatten().mul(J)
        edge_index = edge_index.repeat(1, N) + batch_offset

        # Modify graph.
        edge_index = to_undirected(edge_index)
        edge_index, _ = add_self_loops(edge_index)

        return edge_index

    def _to_graph_seq(self, X):
        N, L, D = X.size()

        return X.reshape(N, L, SMPL_NR_JOINTS, D // SMPL_NR_JOINTS)

    def _from_graph_seq(self, X):
        L, _, A = X.size()

        X = X.reshape(L, -1, SMPL_NR_JOINTS, A)
        X = X.permute(1, 0, 2, 3)

        return X.reshape(-1, L, SMPL_NR_JOINTS * A)

    def batchify(self, X):
        # Reshape for graph.
        X = self._to_graph_seq(X)

        # Get dimension.
        N, L, J, A = X.size()

        # Reshape for temporal batching.
        X = X.permute(1, 0, 2, 3)
        X = X.reshape(L, N * J, A)

        # Calculate batches.
        edge_index = self._get_edge_index(N, J).to(self.device)
        data_list = [Data(X[t], edge_index) for t in range(L)]

        return Batch.from_data_list(data_list)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        # Batch source and tgt sequences by time.
        srcb = self.batchify(src)
        tgtb = self.batchify(tgt)

        # Iterate over batches.
        for t in range(srcb.num_graphs):
            if not t:
                H, C = None, None

            out, (H, C) = self.enc(srcb[t].x, srcb[t].edge_index, H=H, C=C)

        # Prepare output seq.
        outs = torch.zeros(
            tgtb.num_graphs if max_len is None else max_len, *tgtb[0].x.size()
        ).to(self.device)

        # Generate sequence.
        for t in range(outs.size(0)):
            if not t or np.random.random() < teacher_forcing_ratio:
                seed = tgtb[t]
            else:
                seed = Data(out, tgtb[0].edge_index)

            # Pass through networks.
            out, (H, C) = self.dec(seed.x, seed.edge_index, H=H, C=C)
            out = self.post(out)

            outs[0, ...] = out

        return self._from_graph_seq(outs)