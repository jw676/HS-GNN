import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
#from torch_geometric.nn import GINConv, GCNConv, GATConv

import dgl
from dgl.nn.pytorch.conv import GINConv, GATConv
import dgl.function as fn
from dgl.nn.functional import edge_softmax


def gin_mlp_factory(in_dim, out_dim):
    '''
    use the default GIN inside NN from GIN paper, could change to others as a hyper parameter
    :param in_dim:
    :param out_dim:
    :return: GIN NN
    '''
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LeakyReLU(),
        nn.Linear(in_dim, out_dim)
    )


class MLP_Layer(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 out_dim,
                 hid_dim=None,
                 batch_norm=True):
        '''

        :param num_layers:
        :param in_dim:
        :param out_dim:
        :param hid_dim:
        :param batch_norm:
        :return:
        '''
        super().__init__()
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        dims = [in_dim] + [hid_dim for _ in range(num_layers-1)] + [out_dim]

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.linears.append(nn.Linear(d_in, d_out))

        if batch_norm:
            for _ in range(num_layers-1):
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))

        self.relu = nn.ReLU()

        self.norm = batch_norm

    def forward(self, h):

        h = self.linears[0](h)

        if self.norm:
            for batch_norm, linear in zip(self.batch_norms, self.linears[1:]):
                h = self.relu(h)
                h = batch_norm(h)
                h = linear(h)

        else:
            for linear in self.linears[1:]:
                h = self.relu(h)
                h = linear(h)

        return h

class BottomEncoder(nn.Module):
    '''
    DGL version of Layer 1 encoder
    '''
    # todo: heterogeneous version (multiple edge type)
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 num_gin: int,
                 num_gat: int,
                 num_mlp_layers: int,
                 init_eps: float = 0.,
                 learn_eps: bool =True,
                 gin_agg: str ='mean',
                 num_heads: int =8,
                 out_type: str ='concat',
                 device: torch.device =torch.device('cpu')
                 ):
        '''

        :param in_dim: input feature dimension
        :param hid_dim: hidden layer dimension
        :param out_dim: output layer dimension
        :param num_gin: p, number of GIN layers
        :param num_gat: q, number of GAT layers
        :param num_mlp_layers: number of linear layers of MLP inside any GIN layer
        :param init_eps: gin conv epsilon
        :param learn_eps: gin conv epsilon
        :param gin_agg: gin conv aggregation type
        :param num_heads: number of attention heads in gat layer
        :param out_type: how to output Layer 1 encoder, support three type
                         1. <final> output the final layer
                         2. <mean> output the mean of all layers (input, hidden and final)
                         3. <concat> output the concatenation of all layers (input, hidden and final)
        :param device: which device to run the model
        '''

        super().__init__()

        self.out_type = out_type

        if out_type not in ('mean', 'concat', 'final'):
            raise KeyError('Not supported output type {}'.format(out_type))

        self.GINs = nn.ModuleList()

        ginDims = [in_dim] + [hid_dim] * num_gin

        for dim1, dim2 in zip(ginDims[:-1], ginDims[1:]):
            linear = MLP_Layer(num_layers=num_mlp_layers, in_dim=dim1, out_dim=dim2, hid_dim=dim2)
            self.GINs.append(
                GINConv(
                    apply_func=linear,
                    aggregator_type=gin_agg,
                    init_eps=init_eps,
                    learn_eps=learn_eps,
                    activation=nn.ReLU()
                )
            )


        # any gat layer output, do mean aggregation over different heads
        self.GATs = nn.ModuleList()

        gatDims = [hid_dim] * num_gat + [out_dim]

        for dim1, dim2 in zip(gatDims[:-1], gatDims[1:]):
            self.GATs.append(
                GATConv(
                    in_feats=dim1,
                    out_feats=dim2,
                    num_heads=num_heads,
                    negative_slope=0.2,
                    activation=nn.ReLU(),
                    allow_zero_in_degree=False # we do not have zero in-degree node
                )
            )

        self.device = device

    def forward(self, g, h):
        output = [h]
        for gin in self.GINs:
            output.append(gin(g, output[-1]))

        for gat in self.GATs:
            tmp = gat(g, output[-1]) # [𝑁, 𝐻, 𝐷𝑜𝑢𝑡]
            output.append(tmp.mean(dim=1))

        if self.out_type=='mean':
            return torch.cat([x.unsqueeze(-1) for x in output[1:]], dim=-1).mean(-1)

        elif self.out_type=='concat':
            return torch.cat(output[1:], dim=-1)

        else: # final
            return output[-1]


class Predictor(nn.Module):

    def __init__(self,
                 bottom_edge_types,
                 node_types,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_gin,
                 num_gat,
                 num_mlp_layers_gin,
                 init_eps_gin,
                 learn_eps_gin,
                 gin_agg,
                 num_heads_bt,
                 out_type_bt,
                 pca_dim,
                 pi_dim,
                 num_mlp_layers_att,
                 num_att_convs,
                 num_heads_att,
                 out_type_att
                 ):
        super(Predictor, self).__init__()

        self.node_types = node_types

        self.botEncoders = nn.ModuleList()
        self.bottom_edges = bottom_edge_types
        for edge in bottom_edge_types:
            self.botEncoders.append(
                BottomEncoder(
                    in_dim=in_dim,
                    hid_dim=hid_dim,
                    out_dim=hid_dim,
                    num_gin=num_gin,
                    num_gat=num_gat,
                    num_mlp_layers=num_mlp_layers_gin,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    gin_agg=gin_agg,
                    num_heads=num_heads_bt,
                    out_type=out_type_bt
                )
            )

        self.hierarchical2 = HierarchicalLayer(
            pca_dim=pca_dim,
            pi_dim=pi_dim,
            hid_dim=hid_dim,
            num_mlp_layers=num_mlp_layers_att,
            num_atts=num_att_convs,
            num_heads=num_heads_att,
            out_type=out_type_att
        )

        self.hierarchical3 = HierarchicalLayer(
            pca_dim=pca_dim,
            pi_dim=pi_dim,
            hid_dim=hid_dim,
            num_mlp_layers=num_mlp_layers_att,
            num_atts=num_att_convs,
            num_heads=num_heads_att,
            out_type=out_type_att)

        self.relu = nn.ReLU()

        if out_type_bt == 'concat':
            bt_dim = (num_gat + num_gin) * hid_dim # ignore the input features (h_0)
        else:
            bt_dim = hid_dim

        self.linear_bt = nn.Linear(4 * bt_dim, hid_dim)

        if out_type_att == 'concat':
            hie_dim = (num_att_convs + 1) * hid_dim # care about the input hidden representation
        else:
            hie_dim = hid_dim

        self.linear_h2 = nn.Linear(hie_dim, hid_dim)

        self.linear_out = nn.Linear(hie_dim, out_dim)

        self.mse = nn.MSELoss()


    def forward(self, graph):

        with graph.local_scope():

            atom_feats = graph.nodes['atom'].data['coords']

            h = []

            for edge_type, encoder in zip(self.bottom_edges, self.botEncoders):
                subgraph = graph[('atom', edge_type, 'atom')]
                output = encoder(subgraph, atom_feats)
                h.append(output)

            h = torch.cat(h, dim=1)

            h = self.relu(self.linear_bt(h))

            # max pooling: from atom to cluster2
            graph.nodes['atom'].data['h'] = h
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='groupsin1')
            h = graph.nodes['cluster2'].data.pop('h')

            subgraph = graph[('cluster2', 'intersects1', 'cluster2')]
            h = self.hierarchical2(subgraph, h)

            h = self.linear_h2(h)


            # max pooling: from cluster2 to cluster3
            graph.nodes['cluster2'].data['h'] = h
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='groupsin2')
            h = graph.nodes['cluster3'].data.pop('h')

            subgraph = graph[('cluster3', 'intersects3', 'cluster3')]
            h = self.hierarchical3(subgraph, h)

            # todo: add global features
            
            h = self.linear_out(h)

            return h

    def loss(self, graph, target):
        predict = self.forward(graph)

        return self.mse(predict, target)


class HierarchicalLayer(nn.Module):

    def __init__(self,
                 pca_dim: int = 9,
                 pi_dim: int = 64,
                 hid_dim: int = 64,
                 num_mlp_layers: int = 3,
                 num_atts: int = 5,
                 num_heads: int = 4,
                 out_type: str ='concat'
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

            linear = nn.Linear(2 * hid_dim * num_heads, hid_dim)

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

            h = [h]

            for att, linear in zip(self.atts, self.linears):
                x = att(graph, h[-1]).view(x.size(0), -1) # [N, 2 x num_heads x out_dim]
                x = self.relu(linear(x))
                h.append(x)

            if self.out_type == 'mean':
                return torch.cat([x.unsqueeze(-1) for x in h], dim=-1).mean(-1)

            elif self.out_type == 'concat':
                return torch.cat(h, dim=-1)

            else:  # final
                return h[-1]


class Edge_attention(nn.Module):
    def __init__(self, pca_dim, pi_dim, hid_dim, num_mlp_layers, num_heads):
        '''

        :param pca_dim:
        :param pi_dim:
        :param hid_dim:
        :param num_mlp_layers:
        '''

        super(Edge_attention, self).__init__()

        self.pca_attention = MLP_Layer(num_layers=num_mlp_layers,
                                       in_dim=pca_dim * 2,
                                       out_dim=num_heads,
                                       hid_dim=hid_dim,
                                       batch_norm=False)

        self.pi_attention = MLP_Layer(num_layers=num_mlp_layers,
                                      in_dim=pi_dim * 2,
                                      out_dim=num_heads,
                                      hid_dim=hid_dim,
                                      batch_norm=False)

        self.num_heads = num_heads

    def forward(self, graph, h):
        '''

        :param graph:
        :param h: node level feats, [num_of_nodes, hid_dim]
        :return:  [num_of_nodes, 2 * hid_dim]
        '''

        with graph.local_scope():

            graph.ndata['h'] = h.unsqueeze(1).repeat(1, self.num_heads, 1)  # [N, num_heads, out_dim]

            # edge attention over two source of information
            graph.apply_edges(lambda edges: {'h_pca': torch.cat([edges.src['pca'], edges.dst['pca']], dim=1)})

            pcascore = self.leayrelu(self.pca_attention(graph.edata.pop('h_pca'))).unsqueeze(-1) # [E, num_heads, 1]

            graph.apply_edges(lambda edges: {'h_pimg': torch.cat([edges.src['pimg'], edges.dst['pimg']], dim=1)})

            pimgscore = self.leayrelu(self.pi_attention(graph.edata.pop('h_pimg'))).unsqueeze(-1)

            graph.edata['pca'] = edge_softmax(graph, pcascore) # [E, num_heads, 1]

            graph.edata['pimg'] = edge_softmax(graph, pimgscore)

            graph.update_all(fn.u_mul_e('h', 'pca', 'm'), fn.sum('m', 'pca')) # [N, num_heads, out_dim]

            graph.update_all(fn.u_mul_e('h', 'pimg', 'm'), fn.sum('m', 'pimg'))

            output = torch.cat((graph.ndata['pca'], graph.ndata['pimg']), dim=-1)

            return output


class Encoder_L1(nn.Module):
    def __init__(self, num_input, num_hidden, num_heads, p, q, eps=None, concat=True, device=torch.device('cuda:0')):
        '''

        :param num_input: input node feature dimension
        :param num_hidden: hidden representation dimension (also the output dimension of Layer 1 encoder)
        :param num_heads: number of attention heads in GAT layer
        :param p: number of GIN layers
        :param q: number of GAT layers (after GIN)
        :param concat: whether to concatenate all hidden representations as final representation

        '''

        # todo 1: whether to add bns as a hyper parameter, and should it also be applied to GAT?

        # todo 2: nn params initialization

        self.GINconvs = nn.ModuleList()
        self.bns = nn.ModuleList() # batch norm
        GINdims = [num_input] + [num_hidden] * p
        for in_dim, out_dim in zip(GINdims[:-1], GINdims[1:]):
            nnLayer = gin_mlp_factory(in_dim, out_dim).to(device)
            if not eps is None:
                self.GINconvs.append(GINConv(nnLayer, eps).to(device))
            else:
                self.GINconvs.append(GINConv(nnLayer, train_eps=True).to(device))
            self.bns.append(nn.BatchNorm1d(out_dim))

        self.GATconvs = nn.ModuleList()
        for _ in range(q):
            self.GATconvs.append(GATConv(num_hidden, num_hidden, heads=num_heads, concat=False).to(device))

        self.concat = concat
        self.device = device
        self.act = nn.LeakyReLU()

    def forward(self, data):

        # todo 1: should implement a batch version with torch_geometric, or could consider dgl for easy batch access

        # todo 2: whether concatenate all hidden representations?

        features, edgelist = data
        if not isinstance(edgelist, torch.LongTensor):
            edgelist = torch.LongTensor(edgelist).to(self.device)
        # suppose features, edgelist are in good data types

        z = [features]

        for conv, bn in zip(self.GINconvs, self.bns):
            x = conv(z[-1], edgelist)
            x = bn(x)
            x = self.act(x)
            z.append(x)

        for conv in self.GATconvs:
            x = conv(z[-1], edgelist)
            x = self.act(x)
            z.append(x)

        if self.concat:
            x = torch.cat(z[1:], dim=1) # input features may have different dims with convs
        else:
            x = z[-1]

        return x






