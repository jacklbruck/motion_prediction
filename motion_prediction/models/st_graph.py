import torch.nn as nn
import torch

from fairmotion.data.amass_dip import SMPL_JOINT_MAPPING, SMPL_PARENTS, SMPL_NR_JOINTS
from torch_geometric_temporal.nn.attention import ASTGCN

class SpatioTemporalGraphModel(nn.Module):
    def __init__(self):
        super(SpatioTemporalGraphModel, self).__init__()
        # Set paramaters.
        self.device = "cpu"
        self.edge_index = self.get_edge_index()
        #_, _, num_nodes, in_channels = self.to_graph_seq(src).size()

        self.layer = ASTGCN(
            nb_block=3,
            in_channels=3,
            K=3,
            nb_chev_filter=3,
            nb_time_filter=3,
            time_strides=3,
            num_for_predict=24,
            len_input=120,
            num_of_vertices=24
        )

    def to_graph_seq(self, X):
        N, L, D = X.size()

        return X.reshape(N, L, SMPL_NR_JOINTS, D // SMPL_NR_JOINTS)

    def init_weights(self):
        pass

    def from_graph_seq(self, X):
        N, L, _, _ = X.size()

        return X.reshape(N, L, -1)

    def get_edge_index(self):
        return torch.LongTensor(
            [(v, u) for u, v in enumerate(SMPL_PARENTS) if v >= 0]
        ).T

    def forward(self, src, tgt, max_len=24, teacher_forcing_ratio=0.5):
        src = self.to_graph_seq(src)

        X = self.layer(src.permute(0, 2, 3, 1), self.edge_index)

        return self.from_graph_seq(X)