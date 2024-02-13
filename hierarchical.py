import torch
import torch.nn as nn
import dgl

from edgeattn import Edge_attention

class HierarchicalLayer(nn.Module):

    def __init__(self,
                 pca_dim: int = 9,
                 pi_dim: int = 64,
                 hid_dim: int = 64,
                 num_mlp_layers: int = 3,
                 num_atts: int = 5,
                 num_heads: int = 4,
                 out_type: str = 'concat'
                 ):
        super().__init__()
        # todo : multi-head attention
        '''
        self.graph = graph

        pca_dim = self.graph.ndata['pca'].shape[1]
        pi_dim = self.graph.ndata['pi'].shape[1]
        '''

        self.out_type = out_type

        if out_type not in ('mean', 'concat', 'final'):
            raise KeyError('Not supported output type {}'.format(out_type))

        self.atts = nn.ModuleList()
        self.linears = nn.ModuleList()

        for i in range(num_atts):
            att = Edge_attention(
                pca_dim=pca_dim,
                pi_dim=pi_dim,
                hid_dim=hid_dim,
                num_mlp_layers=num_mlp_layers,
                num_heads=num_heads
            )
            if pi_dim != 0:
                linear = nn.Linear(2 * hid_dim * num_heads, hid_dim)
            else:
                linear = nn.Linear(hid_dim * num_heads, hid_dim)

            self.atts.append(att)
            self.linears.append(linear)

        self.relu = nn.ReLU()

    def forward(self, graph, h):
        '''

        :param graph: a heterogeneous graph at Hierarchical Layer,
                      with two tuple relations : <atom, groupsin, cluster>, <cluster, intersects, cluster>
        :param h: learned representation matrix from previous Hierarchical Layer, with shape [num_of_atoms, hid_dim]
        :return: representation matrix for each super node (a.k.a. cluster), with shape [num_of_clusters, 2 * hid_dim]
        '''

        with graph.local_scope():

            graph = dgl.add_self_loop(graph) # in case some nodes are isolated (degree=0)

            num_nodes = h.shape[0]
            h = [h]

            for att, linear in zip(self.atts, self.linears):
                x = att(graph, h[-1]).view(num_nodes, -1)  # [N, 2 x num_heads x hid_dim]
                x = self.relu(linear(x))
                h.append(x)

            if self.out_type == 'mean':
                return torch.cat([x.unsqueeze(-1) for x in h], dim=-1).mean(-1)

            elif self.out_type == 'concat':
                return torch.cat(h, dim=-1)

            else:  # final
                return h[-1]
