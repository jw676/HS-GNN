import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GINConv, GATConv

from nnutilities import MLP_Layer


class BottomEncoder(nn.Module):
    '''
    DGL version of Layer 1 encoder
    '''
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
                    activation=nn.ReLU() # todo: check this activation func
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
                    allow_zero_in_degree=True # we do have some zero in-degree nodes (after cut the boundaries)
                )
            )


    def forward(self, g, h):
        with g.local_scope():

            g = dgl.add_self_loop(g)
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
