import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.functional import edge_softmax

from nnutilities import MLP_Layer



class Edge_attention(nn.Module):
    def __init__(self,
                 pca_dim,
                 pi_dim,
                 hid_dim,
                 num_mlp_layers,
                 num_heads
                 ):
        '''

        :param pca_dim:
        :param pi_dim:
        :param hid_dim:
        :param num_mlp_layers:
        :param num_heads:
        '''

        super(Edge_attention, self).__init__()

        self.pca_attention = MLP_Layer(num_layers=num_mlp_layers,
                                       in_dim=pca_dim * 2,
                                       out_dim=num_heads,
                                       hid_dim=hid_dim,
                                       batch_norm=False)
        if pi_dim != 0: # sometimes we do want to add a persistent image
            self.pi_attention = MLP_Layer(num_layers=num_mlp_layers,
                                          in_dim=pi_dim * 2,
                                          out_dim=num_heads,
                                          hid_dim=hid_dim,
                                          batch_norm=False)

        self.num_heads = num_heads
        self.pi_dim = pi_dim
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, graph, h):
        '''

        :param graph:
        :param h: node level feats, [num_of_nodes, hid_dim]
        :return:  [num_of_nodes, 2 * hid_dim]
        '''

        with graph.local_scope():

            graph.ndata['h'] = h.unsqueeze(1).repeat(1, self.num_heads, 1)  # [N, num_heads, out_dim]

            #graph.ndata['pimg'] = graph.ndata['pimg'].reshape(-1, self.pi_dim)

            # edge attention over two source of information
            graph.apply_edges(lambda edges: {'h_pca': torch.cat([edges.src['pca'], edges.dst['pca']], dim=1)})

            pcascore = self.leakyrelu(self.pca_attention(graph.edata.pop('h_pca'))).unsqueeze(-1) # [E, num_heads, 1]

            graph.edata['pca'] = edge_softmax(graph, pcascore) # [E, num_heads, 1]

            graph.update_all(fn.u_mul_e('h', 'pca', 'm'), fn.sum('m', 'pca')) # [N, num_heads, out_dim]

            output = graph.ndata['pca']

            if self.pi_dim != 0:
                graph.apply_edges(lambda edges: {'h_pimg': torch.cat([edges.src['pimg'], edges.dst['pimg']], dim=1)})

                pimgscore = self.leakyrelu(self.pi_attention(graph.edata.pop('h_pimg'))).unsqueeze(-1)

                graph.edata['pimg'] = edge_softmax(graph, pimgscore)

                graph.update_all(fn.u_mul_e('h', 'pimg', 'm'), fn.sum('m', 'pimg'))

                output = torch.cat((graph.ndata['pca'], graph.ndata['pimg']), dim=-1)

            return output
