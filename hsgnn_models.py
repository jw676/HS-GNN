import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import math

from dgl.nn.pytorch.conv import GINConv, GATConv
from nnutilities import MLP_Layer
from pe import PositionalEncoder


class MLP(nn.Module):
    def __init__(self,
                 in_dim=8,
                 hid_dim=32,
                 out_dim=1,
                 num_layers=4):
        super().__init__()

        self.fcn = MLP_Layer(
            num_layers=num_layers,
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            batch_norm=False
        )

    def forward(self, data):
        return self.fcn(data)


class BTGINs(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim,
                 ):
        super().__init__()


        linear1 = MLP_Layer(
            num_layers=2,
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            batch_norm=True
        )

        self.gin1 = GINConv(
            apply_func=linear1,
            aggregator_type='sum',
            init_eps=0.1,
            learn_eps=True,
            activation=nn.ReLU()
        )

        linear2 = MLP_Layer(
            num_layers=2,
            in_dim=hid_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            batch_norm=True
        )

        self.gin2 = GINConv(
            apply_func=linear2,
            aggregator_type='sum',
            init_eps=0.1,
            learn_eps=True,
            activation=None
        )


    def forward(self, graph, data):

        h = self.gin1(graph, data)

        h = self.gin2(graph, h)

        return h

class BTGNNs(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_heads,
                 ):
        super(BTGNNs, self).__init__()

        if (hid_dim % num_heads) != 0:
            raise AttributeError('hidden dim should be multuplier of num heads')

        self.gat1 = GATConv(
            in_feats=in_dim,
            out_feats=hid_dim // num_heads,
            num_heads=num_heads,
            allow_zero_in_degree=True,
            activation=nn.ReLU()
        )

        self.gat2 = GATConv(
            in_feats=hid_dim,
            out_feats=hid_dim // num_heads,
            num_heads=num_heads,
            allow_zero_in_degree=True,
            activation=nn.ReLU()
        )


    def forward(self, graph, data):

        num_nodes = data.size(0)

        h = self.gat1(graph, data).view(num_nodes, -1) #[N, hid]

        h = self.gat2(graph, h).view(num_nodes, -1) #[N, hid]

        return h


class GIN(nn.Module):

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
                 att_type,
                 final_readout
                 ):
        super().__init__()


        self.botEncoders = nn.ModuleList()
        self.bottom_edges = bottom_edge_types
        for edge in bottom_edge_types:
            self.botEncoders.append(
                BTGINs(
                    in_dim,
                    hid_dim
                )
            )

        self.out_layers = MLP_Layer(
            num_layers=3,
            in_dim=2 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout



    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            atom_feats = graph.nodes['A'].data['feats']

            h = []

            for edge_type, encoder in zip(self.bottom_edges, self.botEncoders):
                subgraph = graph[('A', edge_type, 'A')]
                output = encoder(subgraph, atom_feats)
                h.append(output)

            h = torch.cat(h, dim=-1)

            graph.nodes['A'].data['h'] = h

            # readout
            h = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')

            h = self.out_layers(h)

            return h

class GAT(nn.Module):

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
                 att_type,
                 final_readout
                 ):
        super().__init__()


        self.botEncoders = nn.ModuleList()
        self.bottom_edges = bottom_edge_types
        for edge in bottom_edge_types:
            self.botEncoders.append(
                BTGNNs(
                    in_dim,
                    hid_dim,
                    num_heads_bt
                )
            )

        self.out_layers = MLP_Layer(
            num_layers=3,
            in_dim=len(bottom_edge_types)*hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout



    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            atom_feats = graph.nodes['A'].data['feats']

            h = []

            for edge_type, encoder in zip(self.bottom_edges, self.botEncoders):
                subgraph = graph[('A', edge_type, 'A')]
                output = encoder(subgraph, atom_feats)
                h.append(output)

            h = torch.cat(h, dim=1)

            graph.nodes['A'].data['h'] = h

            # readout
            h = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')

            h = self.out_layers(h)

            return h


class BTEC(nn.Module):

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
                 att_type,
                 final_readout
                 ):
        super().__init__()


        self.botEncoders = nn.ModuleList()
        self.bottom_edges = bottom_edge_types
        for edge in bottom_edge_types:
            botec = nn.ModuleList([
                BTGNNs(
                    in_dim,
                    hid_dim,
                    num_heads_bt
                ),
                BTGINs(
                    hid_dim,
                    hid_dim
                )
            ])
            self.botEncoders.append(botec)


        self.out_layers = MLP_Layer(
            num_layers=3,
            in_dim=len(bottom_edge_types)*hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.relu = nn.ReLU()

        self.final_readout = final_readout

        self.num_input = in_dim


    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            atom_feats = graph.nodes['A'].data['feats'][:, :self.num_input]

            h = []

            for edge_type, encoder in zip(self.bottom_edges, self.botEncoders):
                subgraph = graph[('A', edge_type, 'A')]
                hid_rep = atom_feats
                for gnn in encoder:
                    hid_rep = self.relu(gnn(subgraph, hid_rep))
                h.append(hid_rep)

            h = torch.cat(h, dim=1)

            graph.nodes['A'].data['h'] = h

            # readout
            h = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')

            h = self.out_layers(h)

            return h



class Bottom(nn.Module):

    def __init__(self,
                 bottom_edge_types,
                 in_dim,
                 hid_dim,
                 num_gin,
                 num_gat,
                 init_eps_gin,
                 learn_eps_gin,
                 gin_agg,
                 num_heads_bt,
                 ):
        super().__init__()


        self.botEncoders = nn.ModuleList()
        self.bottom_edges = bottom_edge_types
        for edge in bottom_edge_types:
            botec = nn.ModuleList()

            for i in range(num_gat):
                botec.append(
                    GATConv(
                    in_feats=in_dim if i==0 else hid_dim,
                    out_feats=hid_dim // num_heads_bt,
                    num_heads=num_heads_bt,
                    allow_zero_in_degree=True,
                    activation=nn.ReLU()
                ))

            for i in range(num_gin):

                fcn = MLP_Layer(
                    in_dim=hid_dim,
                    hid_dim=hid_dim,
                    out_dim=hid_dim,
                    batch_norm=True,
                    num_layers=2
                )

                botec.append(
                    GINConv(
                        apply_func=fcn,
                        aggregator_type=gin_agg,
                        init_eps=init_eps_gin,
                        learn_eps=learn_eps_gin,
                        activation=nn.ReLU()
                    )
                )

            self.botEncoders.append(botec)



        self.relu = nn.ReLU()



    def forward(self, graph):

        with graph.local_scope():

            atom_feats = graph.nodes['A'].data['feats']

            h = []

            for edge_type, encoder in zip(self.bottom_edges, self.botEncoders):
                subgraph = graph[('A', edge_type, 'A')]
                hid_rep = atom_feats
                for gnn in encoder:
                    hid_rep = self.relu(gnn(subgraph, hid_rep)).view(atom_feats.size(0), -1)
                h.append(hid_rep)

            h = torch.cat(h, dim=1)

            return h



class BottomSkip(nn.Module):

    def __init__(self,
                 bottom_edge_types,
                 in_dim,
                 hid_dim,
                 num_gin,
                 num_gat,
                 init_eps_gin,
                 learn_eps_gin,
                 gin_agg,
                 num_heads_bt,
                 ):
        super().__init__()


        self.botEncoders = nn.ModuleList()
        self.bottom_edges = bottom_edge_types
        for edge in bottom_edge_types:
            botec = nn.ModuleList()

            for i in range(num_gat):
                botec.append(
                    GATConv(
                    in_feats=in_dim if i==0 else hid_dim,
                    out_feats=hid_dim // num_heads_bt,
                    num_heads=num_heads_bt,
                    allow_zero_in_degree=True,
                    activation=nn.ReLU()
                ))

            for i in range(num_gin):

                fcn = MLP_Layer(
                    in_dim=hid_dim + in_dim if i==0 else hid_dim,
                    hid_dim=hid_dim,
                    out_dim=hid_dim,
                    batch_norm=True,
                    num_layers=2
                )

                botec.append(
                    GINConv(
                        apply_func=fcn,
                        aggregator_type=gin_agg,
                        init_eps=init_eps_gin,
                        learn_eps=learn_eps_gin,
                        activation=nn.ReLU()
                    )
                )

            self.botEncoders.append(botec)

        self.skip = num_gat

        self.relu = nn.ReLU()



    def forward(self, graph, useh=False):

        with graph.local_scope():

            if not useh:
                atom_feats = graph.nodes['A'].data['feats']
            else:
                atom_feats = graph.ndata['h']

            h = []

            for edge_type, encoder in zip(self.bottom_edges, self.botEncoders):
                if not useh:
                    subgraph = graph[('A', edge_type, 'A')]
                else: subgraph = graph
                hid_rep = atom_feats
                for i, gnn in enumerate(encoder):
                    if i==self.skip:
                        hid_rep = torch.cat([hid_rep, atom_feats], -1)
                    hid_rep = self.relu(gnn(subgraph, hid_rep)).view(atom_feats.size(0), -1)
                h.append(hid_rep)

            h = torch.cat(h, dim=1)

            return h


class BEP(nn.Module):

    def __init__(self,
                 bottom_edge_types,
                 node_types,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_gin,
                 num_gat,
                 num_mlp_layers_gin,
                 num_heads_bt,
                 init_eps_gin,
                 learn_eps_gin,
                 gin_agg,
                 out_type_bt,
                 pca_dim,
                 pi_dim,
                 num_mlp_layers_att,
                 num_att_convs,
                 num_heads_att,
                 out_type_att,
                 att_type,
                 glb_dim,
                 final_readout,
        ):
        super().__init__()


        self.botEncoder = BottomSkip(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gin=num_gin,
            num_gat=num_gat,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.fcl = MLP_Layer(
            num_layers=2,
            in_dim=hid_dim*2,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False
        )

        self.in_dim = in_dim

        self.final_readout = final_readout

    def forward(self, graph):

        with graph.local_scope():

            if self.in_dim == 4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][..., :4]

            h = self.botEncoder(graph)

            graph.nodes['A'].data['h'] = h

            h = dgl.readout_nodes(graph, 'h', ntype='A', op=self.final_readout)

            return self.fcl(h)



class H2(nn.Module):

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
                 att_type,
                 final_readout
                 ):
        super().__init__()

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        num_bt_output = len(bottom_edge_types)

        self.H2 = nn.ModuleList()

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * num_bt_output if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )


        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= (1 + num_bt_output) * hid_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            batch_norm=False)

        self.final_readout = final_readout

        self.in_dim = in_dim



    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][..., :self.in_dim]

            h = self.botEncoder(graph) # [N, 2* hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')


            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h

class H2_(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type[:3]=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        self.glbh = True if glb_dim > 0 else False

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * len(bottom_edge_types) + add_dim + glb_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= (len(bottom_edge_types)+1) * hid_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            batch_norm=False)

        self.final_readout = final_readout

        self.pi_dim = pi_dim


    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :self.input]

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pcavec':
                add = graph.nodes['C2'].data['pca'][..., :6].view(h.size(0), -1) # todo: 0:6 vectors, 6:8 value ratio, 8:10 values
            elif self.add_type == 'pcaval':
                add = graph.nodes['C2'].data['pca'][..., -2:].view(h.size(0), -1) # todo: 0:6 vectors, 6:8 value ratio, 8:10 values
            elif self.add_type == 'pcarat':
                add = graph.nodes['C2'].data['pca'][..., 6:8].view(h.size(0), -1) # todo: 0:6 vectors, 6:8 value ratio, 8:10 values

            h = torch.cat([h, add], dim=-1)

            if self.glbh:
                glbh = graph.nodes['C2'].data['feats']
                h = torch.cat([h, glbh], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h

    def get_hidden(self, graph, glbfeats=None):

        with graph.local_scope():

            graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :self.input]

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pcavec':
                add = graph.nodes['C2'].data['pca'][..., :6].view(h.size(0), -1) # todo: 0:6 vectors, 6:8 value ratio, 8:10 values
            elif self.add_type == 'pcaval':
                add = graph.nodes['C2'].data['pca'][..., -2:].view(h.size(0), -1) # todo: 0:6 vectors, 6:8 value ratio, 8:10 values
            elif self.add_type == 'pcarat':
                add = graph.nodes['C2'].data['pca'][..., 6:8].view(h.size(0), -1) # todo: 0:6 vectors, 6:8 value ratio, 8:10 values

            h = torch.cat([h, add], dim=-1)

            if self.glbh:
                glbh = graph.nodes['C2'].data['feats']
                h = torch.cat([h, glbh], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h, get_hidden=True)

            return h



class H2_PICA(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        add_dim = pca_dim + pi_dim

        self.add_type = att_type

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * 2 + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 3 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout



    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            if self.input==4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :4]

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            add0 = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            if self.add_type == 'pcavec':
                add = graph.nodes['C2'].data['pca'][..., :6].view(h.size(0), -1) # todo: 0:6 vectors, 6:8 value ratio, 8:10 values
            elif self.add_type == 'pcaval':
                add = graph.nodes['C2'].data['pca'][..., -2:].view(h.size(0), -1) # todo: 0:6 vectors, 6:8 value ratio, 8:10 values
            elif self.add_type == 'pcarat':
                add = graph.nodes['C2'].data['pca'][..., 6:8].view(h.size(0), -1) # todo: 0:6 vectors, 6:8 value ratio, 8:10 values

            h = torch.cat([h, add0, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h

class H2_PE(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.pe = PositionalEncoder(d_input=3, n_freqs=10)

        in_dim = self.pe.d_output + 1

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * 2 + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 3 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout

        self.pi_dim = pi_dim

    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            if self.input==4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :4]

            positembed = self.pe(graph.nodes['A'].data['feats'][:, :3])

            graph.nodes['A'].data['feats'] = torch.cat([positembed, graph.nodes['A'].data['feats'][:, -1:]], dim=-1)

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'][..., :4].view(h.size(0), -1) # todo

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h

class H2_RNI(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.botEncoder = BottomSkip(
            bottom_edge_types=bottom_edge_types,
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * 2 + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        self.input = in_dim
        self.hid_dim = hid_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 3 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout

        self.pi_dim = pi_dim

        self.rni = node_types

    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            if self.input==4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :4]

            # RNI
            h_rni = torch.zeros(graph.num_nodes('A'), self.hid_dim-self.input).to(graph.device)
            if self.rni=='normal':
                h_rni.normal_(0, 1)
            elif self.rni == 'uniform':
                h_rni.uniform_(-1,1)
            else:
                print('Not supported rni')
                exit(0)

            graph.nodes['A'].data['feats'] = torch.cat([graph.nodes['A'].data['feats'], h_rni], dim=-1)

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'][..., :4].view(h.size(0), -1) # todo

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h


class H2_V(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        self.fcC2 = nn.Linear(2*hid_dim + add_dim, hid_dim)

        self.fcV2 = nn.Linear(glb_dim, hid_dim)

        for i in range(num_att_convs):
            '''
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )
            '''

            self.H2.append(
                GATConv(
                    in_feats=hid_dim,
                    num_heads=num_heads_att,
                    out_feats=hid_dim // num_heads_att,
                    activation=nn.ReLU(),
                    allow_zero_in_degree=True
                )
            )


        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 3 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout

        self.pi_dim = pi_dim

        self.relu = nn.ReLU()

    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            h = self.botEncoder(graph) # [N, 2* hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'].view(h.size(0), -1)

            h = torch.cat([h, add], dim=-1)
            h = self.relu(self.fcC2(h)) # C2 initial features [N, hid]
            graph.nodes['C2'].data['h'] = h

            hv = self.relu(self.fcV2(graph.nodes['V2'].data.pop('glb')))
            graph.nodes['V2'].data['h'] = hv

            hgs = []

            for g in dgl.unbatch(graph):
                sg = dgl.edge_type_subgraph(g, [('C2', 'I2', 'C2'), ('C2', 'P2', 'V2')])
                hgs.append(dgl.to_homogeneous(sg, ndata='h'))

            hg = dgl.batch(hgs)

            h = hg.ndata['h']

            for conv in self.H2:
                h = conv(hg, h).view(h.size(0), -1)

            hg.ndata['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(hg, 'h', op=self.final_readout)

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h



class H2_VB(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.bottom_edge_types = bottom_edge_types
        self.botEncoders = nn.ModuleList()
        for bottom_edge_type in bottom_edge_types:
            botEncoder = BottomSkip(
                bottom_edge_types=bottom_edge_type,
                in_dim=in_dim,
                hid_dim=hid_dim,
                num_gat=num_gat,
                num_gin=num_gin,
                init_eps_gin=init_eps_gin,
                learn_eps_gin=learn_eps_gin,
                gin_agg=gin_agg,
                num_heads_bt=num_heads_bt
            )
            self.botEncoders.append(botEncoder)

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        self.fcV1 = nn.Linear(glb_dim, in_dim)

        self.fcC2 = nn.Linear(2*hid_dim + add_dim, hid_dim)


        for i in range(num_att_convs):


            self.H2.append(
                GATConv(
                    in_feats=hid_dim,
                    num_heads=num_heads_att,
                    out_feats=hid_dim // num_heads_att,
                    activation=nn.ReLU(),
                    allow_zero_in_degree=True
                )
            )


        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 3 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout

        self.pi_dim = pi_dim

        self.relu = nn.ReLU()

    def forward(self, graph, glbfeats=None):

        with graph.local_scope():



            graph.nodes['A'].data['h'] = graph.nodes['A'].data['feats']

            graph.nodes['V1'].data['h'] = self.fcl(graph.nodes['V1'].data['feats'])

            bottomH = []

            for etype, botEncoder in zip(self.bottom_edge_types, self.botEncoders):

                hgs = []
                for g in dgl.unbatch(graph):
                    sg = dgl.edge_type_subgraph(g, [('A', etype, 'A'), ('V1', 'P1', 'A')])
                    hgs.append(dgl.to_homogeneous(sg, ndata='h'))

                hg = dgl.batch(hgs)

                bottomH.append(botEncoder(hg, True))

            h = torch.cat(bottomH, dim=-1)

            graph.nodes['A']




            h = self.botEncoder(graph) # [N, 2* hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'].view(h.size(0), -1)

            h = torch.cat([h, add], dim=-1)
            h = self.relu(self.fcC2(h)) # C2 initial features [N, hid]
            graph.nodes['C2'].data['h'] = h

            hv = self.relu(self.fcV2(graph.nodes['V2'].data.pop('glb')))
            graph.nodes['V2'].data['h'] = hv

            hgs = []

            for g in dgl.unbatch(graph):
                sg = dgl.edge_type_subgraph(g, [('C2', 'I2', 'C2'), ('C2', 'P2', 'V2')])
                hgs.append(dgl.to_homogeneous(sg, ndata='h'))

            hg = dgl.batch(hgs)

            h = hg.ndata['h']

            for conv in self.H2:
                h = conv(hg, h).view(h.size(0), -1)

            hg.ndata['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(hg, 'h', op=self.final_readout)

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h




class H2_N(nn.Module):

    '''
    0-layer: add a virtual node T, connecting to all atoms via P (T-P-A), uni-directional, apply a GAT layer to let the atoms learn form the normalized distance (x,y,z)
    bottom: concatenate the learned embedding + additional atom level features like degrees, or possible global ones
    '''

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
                 att_type,
                 final_readout,
                 virt_feats = 5,
                 ):
        super().__init__()

        self.PGat = GATConv(
            in_feats=(virt_feats, 3),
            out_feats=virt_feats,
            num_heads=num_heads_bt,
            allow_zero_in_degree=True
        )

        botin_dim = virt_feats * num_heads_bt + in_dim - 3 # or - virt_feats

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=botin_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * 2 + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 3 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout


    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            sg = dgl.edge_type_subgraph(graph, ('P'))
            #sg = graph['T', 'P', 'A']

            # todo should add a self loop
            #sg = dgl.add_self_loop(sg)

            tfeat = graph.nodes['T'].data['dist']
            acoord = graph.nodes['A'].data['feats'][:, :3]

            h = self.PGat(sg, (tfeat, acoord)).view(graph.num_nodes('A'), -1)   # [N, virt_dim x num_heads]

            graph.nodes['A'].data['feats'] = torch.cat([h, graph.nodes['A'].data['feats'][:, 3:]], dim=-1)

            if self.input == 4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :4]

            h = self.botEncoder(graph)  # [N, 2* hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'].view(h.size(0), -1)

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h


class H2_GAT(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        for i in range(num_att_convs):

            self.H2.append(
                GATConv(
                    in_feats=hid_dim*2 + add_dim if i==0 else hid_dim,
                    num_heads=num_heads_att,
                    out_feats=hid_dim//num_heads_att,
                    allow_zero_in_degree=True,
                    activation=nn.ReLU()
                )

            )

        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 3 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout

        self.pi_dim = pi_dim

    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            if self.input==4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :4]

            h = self.botEncoder(graph) # [N, 2* hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'].view(h.size(0), -1)

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h).view(h.size(0), -1)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h





class H2_CAT(nn.Module):

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
                 att_type,
                 final_readout
                 ):
        super().__init__()

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * 2 + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )


        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 3 * hid_dim + glb_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout


        self.glb_coef = 0.1


    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            h = self.botEncoder(graph) # [N, 2* hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'].view(h.size(0), -1)

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2, self.glb_coef * glbfeats], dim=-1)

            h = self.out_layers(h)

            return h


class H2_EdgeAtt(nn.Module):

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
                 att_type,
                 final_readout
                 ):

        super().__init__()

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.out_layers = MLP_Layer(
            num_layers=3,
            in_dim=3 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout

        # edge attention
        self.num_heads = num_heads_att

        if att_type=='pimg':
            self.att_dim = pi_dim
        elif att_type=='pca':
            self.att_dim = pca_dim
        else:
            raise KeyError('invalid attention type, supported: pimg / pca')

        self.att_type = att_type

        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        self.h2_mlp = MLP_Layer(
            num_layers=2,
            in_dim=2 * hid_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim // num_heads_att,
            batch_norm=False
        )

        self.att_mlp = MLP_Layer(
            num_layers=2,
            in_dim=2 * self.att_dim,
            hid_dim=hid_dim,
            out_dim=num_heads_att,
            batch_norm=False
        )


    def edge_attention(self, graph, h):

        with graph.local_scope():

            h = self.h2_mlp(h)

            graph.ndata['h'] = h.unsqueeze(1).repeat(1, self.num_heads, 1)  # [N, num_heads, out_dim]

            graph.apply_edges(
                lambda edges:
                {
                    'info': torch.cat([edges.src[self.att_type].view(-1, self.att_dim),
                                       edges.dst[self.att_type].view(-1, self.att_dim)], dim=-1)
                }
            )

            # [E, num_heads, 1]
            att_score = self.leakyrelu(self.att_mlp(graph.edata.pop('info'))).unsqueeze(-1)

            graph.edata['att_coef'] = dgl.nn.functional.edge_softmax(graph, att_score)

            graph.update_all(fn.u_mul_e('h', 'att_coef', 'm'), fn.sum('m', 'h'))

            output = graph.ndata['h'].view(h.size(0), -1) # [N, hid_dim]

            return self.relu(output)


    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            h = self.botEncoder(graph)

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            subgraph = graph['C2', 'I2', 'C2']
            h = self.edge_attention(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')

            h = torch.cat([h1, h2], dim=-1)

            h = self.out_layers(h)

            return h




class GAT_N(nn.Module):

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
                 att_type,
                 final_readout
                 ):
        super().__init__()


        self.botEncoders = nn.ModuleList()
        self.bottom_edges = bottom_edge_types
        for edge in bottom_edge_types:
            self.botEncoders.append(
                BTGNNs(
                    in_dim,
                    hid_dim,
                    num_heads_bt
                )
            )

        self.out_layers = MLP_Layer(
            num_layers=3,
            in_dim=2*hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout



    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            atom_feats = graph.nodes['A'].data['feats'][:, :4]

            h = []

            for edge_type, encoder in zip(self.bottom_edges, self.botEncoders):
                subgraph = graph[('A', edge_type, 'A')]
                output = encoder(subgraph, atom_feats)
                h.append(output)

            h = torch.cat(h, dim=1)

            graph.nodes['A'].data['h'] = h

            # readout
            h = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')

            h = self.out_layers(h)

            return h



class H3_(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.botEncoder = Bottom(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        num_bt_output = len(bottom_edge_types)

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * num_bt_output + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        fcl3 = MLP_Layer(
            num_layers=2,
            in_dim=hid_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            batch_norm=True
        )

        self.H3 = GINConv(
            apply_func=fcl3,
            aggregator_type=gin_agg,
            init_eps=init_eps_gin,
            learn_eps=learn_eps_gin,
            activation=nn.ReLU()
        )

        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=3,
            in_dim= (2 + num_bt_output) * hid_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            batch_norm=False)

        self.final_readout = final_readout

        self.pi_dim = pi_dim

    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :self.input]

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pcavec':
                add = graph.nodes['C2'].data['pca'][..., :6].view(h.size(0),
                                                                  -1)  # todo: 0:6 vectors, 6:8 value ratio, 8:10 values
            elif self.add_type == 'pcaval':
                add = graph.nodes['C2'].data['pca'][..., -2:].view(h.size(0),
                                                                   -1)  # todo: 0:6 vectors, 6:8 value ratio, 8:10 values
            elif self.add_type == 'pcarat':
                add = graph.nodes['C2'].data['pca'][..., 6:8].view(h.size(0),
                                                                   -1)  # todo: 0:6 vectors, 6:8 value ratio, 8:10 values

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # H3 layer
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G2')
            h = graph.nodes['C3'].data.pop('h') # [N, hid]

            subgraph = graph['C3', 'I3', 'C3']
            h = self.H3(subgraph, h)

            graph.nodes['C3'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')
            h3 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C3')

            h = torch.cat([h1, h2, h3], dim=-1)

            h = self.out_layers(h)

            return h


class H3_PI(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.botEncoder = BottomSkip(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * 2 + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        self.H3 = nn.ModuleList()

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim + add_dim if i == 0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H3.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )


        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 4 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout

        self.pi_dim = pi_dim

    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            if self.input==4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :4]

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'][..., :4].view(h.size(0), -1) # todo

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # H3 layer
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G2')
            h = graph.nodes['C3'].data.pop('h') # [N, hid]

            if self.add_type == 'pimg':
                add = graph.nodes['C3'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C3'].data['pca'][..., :4].view(h.size(0), -1) # todo

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C3', 'I3', 'C3']

            for conv in self.H3:
                h = conv(subgraph, h)

            graph.nodes['C3'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')
            h3 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C3')

            h = torch.cat([h1, h2, h3], dim=-1)

            h = self.out_layers(h)

            return h


class H3_PCA(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        self.botEncoder = BottomSkip(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type[:3]=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * 2 + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        self.H3 = nn.ModuleList()

        add_dim = 0

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim + add_dim if i == 0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H3.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )


        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= 4 * hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout

        self.pi_dim = pi_dim

    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            if self.input==4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :4]

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type[:3] == 'pca':
                add = graph.nodes['C2'].data['pca'][..., -2:].view(h.size(0), -1) # todo. values

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # H3 layer
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G2')
            h = graph.nodes['C3'].data.pop('h') # [N, hid]
            '''
            if self.add_type == 'pimg':
                add = graph.nodes['C3'].data['pimg'].view(h.size(0), -1)
            elif self.add_type[:3] == 'pca':
                add = graph.nodes['C3'].data['pca'][..., -2:].view(h.size(0), -1) # todo values
            

            h = torch.cat([h, add], dim=-1)
            '''
            subgraph = graph['C3', 'I3', 'C3']

            for conv in self.H3:
                h = conv(subgraph, h)

            graph.nodes['C3'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')
            h3 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C3')

            h = torch.cat([h1, h2, h3], dim=-1)

            h = self.out_layers(h)

            return h

class H3_GLB(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        # todo: skip layer bottom

        self.botEncoder = BottomSkip(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * 2 + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        # todo we add pimg in H3

        fcl3 = MLP_Layer(
            num_layers=2,
            in_dim=hid_dim + add_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            batch_norm=True
        )

        self.H3 = GINConv(
            apply_func=fcl3,
            aggregator_type=gin_agg,
            init_eps=init_eps_gin,
            learn_eps=learn_eps_gin,
            activation=nn.ReLU()
        )

        self.glb_fcl = MLP_Layer(
            num_layers=2,
            in_dim=glb_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            batch_norm=False
        )

        # todo: add attention

        self.attn = nn.Parameter(torch.zeros(5, hid_dim))

        nn.init.kaiming_uniform_(self.attn, a=math.sqrt(5))

        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout


        self.hid_dim = hid_dim

    def forward(self, graph, glbfeats):

        with graph.local_scope():

            if self.input==4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :4]

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'][..., :4].view(h.size(0), -1) # todo

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # H3 layer
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G2')
            h = graph.nodes['C3'].data.pop('h') # [N, hid]

            if self.add_type == 'pimg':
                add = graph.nodes['C3'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C3'].data['pca'][..., :4].view(h.size(0), -1) # todo

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C3', 'I3', 'C3']
            h = self.H3(subgraph, h)

            graph.nodes['C3'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h0, h1 = h1[..., :self.hid_dim], h1[..., self.hid_dim:] # we have two edge types
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')
            h3 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C3')

            h4 = self.glb_fcl(glbfeats)

            h = torch.stack([h0, h1, h2, h3, h4], dim=-1).transpose(-1,-2) # [#Graph, 5, hid]

            attn_score = nn.functional.leaky_relu(torch.sum(h * self.attn, -1))

            attn_coef = nn.functional.softmax(attn_score, -1).unsqueeze(-1).repeat(1, 1, self.hid_dim) # [#Graph, 5, hid]

            h = torch.sum(h * attn_coef, dim=-2) # [#Graph, hid]

            h = self.out_layers(h)

            return h


class H2_GLB(nn.Module):

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
                 att_type,
                 final_readout,
                 ):
        super().__init__()

        # todo: skip layer bottom

        self.botEncoder = BottomSkip(
            bottom_edge_types=bottom_edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=init_eps_gin,
            learn_eps_gin=learn_eps_gin,
            gin_agg=gin_agg,
            num_heads_bt=num_heads_bt
        )

        self.H2 = nn.ModuleList()

        if att_type=='pca':
            add_dim = pca_dim
        elif att_type=='pimg':
            add_dim = pi_dim
        else:
            raise KeyError('pca or pimg')

        self.add_type = att_type

        for i in range(num_att_convs):
            fcn = MLP_Layer(
                num_layers=2,
                in_dim=hid_dim * 2 + add_dim if i==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type=gin_agg,
                    init_eps=init_eps_gin,
                    learn_eps=learn_eps_gin,
                    activation=nn.ReLU()
                )
            )

        self.glb_fcl = MLP_Layer(
            num_layers=2,
            in_dim=glb_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            batch_norm=False
        )

        # todo: add attention

        self.attn = nn.Parameter(torch.zeros(4, hid_dim))

        nn.init.kaiming_uniform_(self.attn, a=math.sqrt(5))

        self.input = in_dim

        self.out_layers = MLP_Layer(
            num_layers=2,
            in_dim= hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            batch_norm=False)

        self.final_readout = final_readout


        self.hid_dim = hid_dim

    def forward(self, graph, glbfeats):

        with graph.local_scope():

            if self.input==4:
                graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :4]

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            if self.add_type == 'pimg':
                add = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)
            elif self.add_type == 'pca':
                add = graph.nodes['C2'].data['pca'][..., :4].view(h.size(0), -1) # todo

            h = torch.cat([h, add], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h


            # readout
            h1 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='A')
            h0, h1 = h1[..., :self.hid_dim], h1[..., self.hid_dim:] # we have two edge types
            h2 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C2')
            #h3 = dgl.readout_nodes(graph, 'h', op=self.final_readout, ntype='C3')

            h4 = self.glb_fcl(glbfeats)

            h = torch.stack([h0, h1, h2, h4], dim=-1).transpose(-1,-2) # [#Graph, 4, hid]

            attn_score = nn.functional.leaky_relu(torch.sum(h * self.attn, -1))

            attn_coef = nn.functional.softmax(attn_score, -1).unsqueeze(-1).repeat(1, 1, self.hid_dim) # [#Graph, 4, hid]

            h = torch.sum(h * attn_coef, dim=-2) # [#Graph, hid]

            h = self.out_layers(h)

            return h
