import torch
import torch.nn as nn
import numpy as np
import sympy as sym
import dgl
import dgl.function as fn

from basis_utils import bessel_basis, real_sph_harm
from bessel_basis import Envelope, BesselBasisLayer
from sperical_besis import SphericalBasisLayer
from embedding import EmbeddingBlock
from interaction import InteractionBlock
from output import OutputBlock

def swish(x):
    """
    Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """
    return x * torch.sigmoid(x)

class DimeNet(nn.Module):
    def __init__(self,
                 emb_size,
                 num_blocks,
                 num_bilinear,
                 num_spherical,
                 num_radial,
                 cutoff=0.5,
                 envelop_exponent=5,
                 num_before_skip=1,
                 num_after_skip=2,
                 num_dense_output=3,
                 num_targets=12,
                 activation=swish,
                 output_init=nn.init.uniform_,
                 device='cuda:0'):
        super(DimeNet, self).__init__()

        self.num_blocks = num_blocks
        self.num_radial = num_radial

        self.rbf_layer = BesselBasisLayer(num_radial=num_radial,
                                          cutoff=cutoff,
                                          envelop_exponent=envelop_exponent,
                                          device=device)


        self.sbf_layer = SphericalBasisLayer(num_spherical=num_spherical,
                                             num_radial=num_radial,
                                             cutoff=cutoff,
                                             envelop_exponent=envelop_exponent)

        self.emb_block = EmbeddingBlock(emb_size=emb_size,
                                        num_radial=num_radial,
                                        bessel_funcs=self.sbf_layer.get_bessel_funcs(),
                                        cutoff=cutoff,
                                        envelop_exponent=envelop_exponent,
                                        activation=activation)



        self.output_blocks = nn.ModuleList({
            OutputBlock(emb_size=emb_size,
                        num_radial=num_radial,
                        num_dense=num_dense_output,
                        num_targets=num_targets,
                        activation=activation,
                        output_init=output_init)
            for _ in range(num_blocks + 1)
        })

        self.interaction_blocks = nn.ModuleList({
            InteractionBlock(emb_size=emb_size,
                             num_radial=num_radial,
                             num_spherical=num_spherical,
                             num_bilinear=num_bilinear,
                             num_before_skip=num_before_skip,
                             num_after_skip=num_after_skip,
                             activation=activation)
            for _ in range(num_blocks)
        })


    def edge_init(self, edges):
        # calculate angles k -> j -> i
        # more intuitive method: compute cosine and then torch.acos()
        R1, R2 = edges.src['o'], edges.dst['o'] # 'o' should be the difference vector
        x = torch.sum(R1 * R2, dim=-1)
        y = torch.cross(R1, R2)
        y = torch.norm(y, dim=-1)
        angle = torch.atan2(y, x)
        # transform via angles
        cbf = [f(angle) for f in self.sbf_layer.get_sph_funcs()]
        cbf = torch.stack(cbf, dim=1) # [None, 7]
        cbf = cbf.repeat_interleave(self.num_radial, dim=1) # [None, 42]
        sbf = edges.src['rbf_env'] * cbf # [none, 42]
        return {'sbf': sbf}

    def forward(self, g, l_g):
        # add rbf features for each edge in one batch graph, [num_radial, ]
        g = self.rbf_layer(g)
        # embedding
        g = self.emb_block(g)
        # output
        P = self.output_blocks[0](g) # [batch_size, num_targets]

        # prepare sbf feature before the following blocks
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g.apply_edges(self.edge_init)
        # interaction
        for i in range(self.num_blocks):
            g = self.interaction_blocks[i](g, l_g)
            #print(g.edata['m'])
            p = self.output_blocks[i + 1](g, True)

            P += p

        return P


