import torch
import torch.nn as nn
import dgl
import dgl.function as fn

from bottom import BottomEncoder
from hierarchical import HierarchicalLayer
from nnutilities import MLP_Layer

class HSGNN(nn.Module):

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
                 out_type_att,
                 glb_dim,
                 final_readout,
                 loss_tp,
                 device
                 ):
        super(HSGNN, self).__init__()

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

        self.mlp_after_bt = MLP_Layer(num_layers=2,
                                      in_dim=len(bottom_edge_types) * bt_dim,
                                      out_dim=hid_dim,
                                      hid_dim=hid_dim,
                                      batch_norm=False)

        if out_type_att == 'concat':
            hie_dim = (num_att_convs + 1) * hid_dim # care about the input hidden representation
        else:
            hie_dim = hid_dim

        self.mlp_after_h2 = MLP_Layer(num_layers=2,
                                      in_dim=hie_dim,
                                      out_dim=hid_dim,
                                      hid_dim=hid_dim,
                                      batch_norm=False)

        self.mlp_after_h3 = MLP_Layer(num_layers=2,
                                      in_dim=hie_dim + glb_dim,
                                      out_dim=out_dim,
                                      hid_dim=hid_dim,
                                      batch_norm=False)

        self.final_readout = final_readout

        self.device = device

        if loss_tp=='mse':
            self.loss_fn = nn.MSELoss()
        elif loss_tp=='l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise KeyError(
                'Not supported Loss type'
            )


    def forward(self, graph, glbfeats):

        #outputs = []

        #for i, graph in enumerate(graphs):

        #    glbfeat = glbfeats[i]

        with graph.local_scope():

            atom_feats = graph.nodes['A'].data['feats']

            h = []

            for edge_type, encoder in zip(self.bottom_edges, self.botEncoders):
                subgraph = graph[('A', edge_type, 'A')]
                output = encoder(subgraph, atom_feats)
                h.append(output)

            h = torch.cat(h, dim=1)

            h = self.relu(self.mlp_after_bt(h))

            # max pooling: from atom to cluster2
            graph.nodes['A'].data['h'] = h
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            subgraph = graph[('C2', 'I2', 'C2')]
            h = self.hierarchical2(subgraph, h)

            h = self.mlp_after_h2(h)


            # max pooling: from cluster2 to cluster3
            graph.nodes['C2'].data['h'] = h
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G2')
            h = graph.nodes['C3'].data.pop('h')

            subgraph = graph[('C3', 'I3', 'C3')]
            h = self.hierarchical3(subgraph, h)

            # readout
            graph.nodes['C3'].data['h'] = h

            h = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C3')

            # add global features (whole bundle)
            if not glbfeats is None:
                h = torch.cat([h, glbfeats], dim=-1)

            h = self.linear_out(h)

        return h


    def loss(self, graph, glbfeats, target):

        h = self.forward(graph, glbfeats).squeeze()
        mse = self.loss_fn(h, target)
        return torch.sqrt(mse / torch.sum(target ** 2))  # prmse

        predict = self.forward(graph, glbfeats)

        mse = self.loss_fn(predict, target.unsqueeze(1))

        return torch.sqrt(mse / torch.sum(target ** 2))  # prmse
