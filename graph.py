import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

import numpy as np
import json
import os

import scipy.sparse as sp

import pandas as pd

import networkx as nx


class dglGraph():
    def __init__(self, filepath, mapping, glbdata):
        self.path = filepath
        self.mapping = mapping
        self.glbdata = glbdata

    def get_glbdata(self, cnt):
        '''
        obtain global data (feats and labels) of a cnt bundle
        global features include:
            '# of Tubes', '# walls per CNT', '# of Atoms',
            '# of Bonds', 'Bond Ratio', 'CNT height', 'Initial Dia (Ang.)',
            , 'Cross-sectional area (nm^2)'
        global pca, global pimgs
        labels include:
            strength and modulus
        :param cnt: index of cnt (int)
        :return: global features, labels
        '''
        data = self.glbdata.loc[self.mapping[str(cnt)]].to_numpy()
        glbfeatures = data[:8].astype(float)
        targets = data[8:].astype(float)

        #glbpca = np.load(os.path.join(self.path+self.mapping[cnt], 'pca_bundle.npy'))

        #glbpimg = np.load(os.path.join(self.path+self.mapping[cnt], 'pimg_bundle.npy'))

        return targets, glbfeatures#, glbpca, glbpimg


    def get_Bottom(self, cnt):
        eB = np.load(os.path.join(cnt, 'eB.npy'))
        #eE = np.load(os.path.join(cnt, 'eE.npy'))
        eD = np.load(os.path.join(cnt, 'eD.npy'))
        #eR = np.load(os.path.join(cnt, 'eR.npy'))
        coords = np.load(os.path.join(cnt, 'coord.npy'))
        g = nx.from_edgelist(eB)

        degrees = []
        for i in range(coords.shape[0]):
            if i not in g.nodes():
                g.add_node(i)
                add_edge = np.array([i, i])
                eB = np.concatenate([eB, add_edge.reshape((1,2))], axis=0)
            degrees.append(nx.degree(g, i))

        return eB, eD, coords, np.array(degrees)
        #return eB, eE, eD, eR, coords

    def get_Hierarchical(self, cnt, layer):
        layer = str(layer)

        adj = sp.load_npz(os.path.join(cnt, 'adjL'+layer+'.npz'))

        cluster = sp.load_npz(os.path.join(cnt, 'clstL'+layer+'.npz'))

        num_super_nodes = adj.shape[0]

        clst_src_nodes = [node for snode in range(num_super_nodes) for node in cluster[snode].nonzero()[1]]
        clst_dst_nodes = [snode for snode in range(num_super_nodes) for _ in cluster[snode].nonzero()[1]]

        # adj is symmetric, no need to add reverse
        adj_src_nodes = [node for i in range(num_super_nodes) for node in adj[i].nonzero()[1]]
        adj_dst_nodes = [i for i in range(num_super_nodes) for _ in adj[i].nonzero()[1]]

        pcafile = 'pcaL' +layer + '.npy'
        pca = np.load(os.path.join(cnt, pcafile))

        pimgfile = 'pimgL' + layer + '.npy'
        pimgs = np.load(os.path.join(cnt, pimgfile))


        return [clst_src_nodes, clst_dst_nodes], [adj_src_nodes, adj_dst_nodes], pca, pimgs

    def get_graph(self, cnt):
        cntfile = os.path.join(os.path.join(self.path, self.mapping[str(cnt)]), 'initial')

        #eB, eE, eD, eR, coords = self.get_Bottom(cntfile)
        eB, eD, coords, degrees = self.get_Bottom(cntfile)

        clst_2, layer_2, pca_2, pimg_2 = self.get_Hierarchical(cntfile, 2)

        clst_3, layer_3, pca_3, pimg_3 = self.get_Hierarchical(cntfile, 3)


        graph_data = {
            ('A', 'B', 'A'): (np.append(eB[:, 0], eB[:, 1]), np.append(eB[:, 1], eB[:, 0])),
            #('atom', 'edist', 'atom'): (np.append(eE[:, 0], eE[:, 1]), np.append(eE[:, 1], eE[:, 0])),
            ('A', 'D', 'A'): (np.append(eD[:, 0], eD[:, 1]), np.append(eD[:, 1], eD[:, 0])),
            #('atom', 'rdist', 'atom'): (np.append(eR[:, 0], eR[:, 1]), np.append(eR[:, 1], eR[:, 0])),
            ('A', 'G1', 'C2'): (clst_2[0], clst_2[1]),
            ('C2', 'I2', 'C2'): (layer_2[0], layer_2[1]),
            ('C2', 'G2', 'C3'): (clst_3[0], clst_3[1]),
            ('C3', 'I3', 'C3'): (layer_3[0], layer_3[1])
            # todo: sometimes there is no I3 edge type
        }

        graph = dgl.heterograph(graph_data)

        feats = np.concatenate([coords, degrees.reshape((-1, 1))], axis=-1)
        graph.nodes['A'].data['feats'] = torch.FloatTensor(feats)
        # todo : coord -> feats (add node degree)
        graph.nodes['C2'].data['pca'] = torch.FloatTensor(pca_2)
        graph.nodes['C2'].data['pimg'] = torch.FloatTensor(pimg_2)
        graph.nodes['C3'].data['pca'] = torch.FloatTensor(pca_3)
        graph.nodes['C3'].data['pimg'] = torch.FloatTensor(pimg_3)

        targets, glbfeatures = self.get_glbdata(cnt)

        return graph, torch.FloatTensor(glbfeatures), torch.FloatTensor(targets)


    def get_write_dataset(self, cntindex):
        graphs = []
        glbfeats = []
        targets = []

        for i, cnt in enumerate(cntindex):
            if i % 100 == 0: print(cnt)
            #print(cnt)
            graph, feats, target = self.get_graph(cnt)
            graphs.append(graph)
            glbfeats.append(feats.view(1, -1))
            targets.append(target.view(1, -1))

        data = {
            'labels': torch.cat(targets, dim=0),
            'feats': torch.cat(glbfeats, dim=0),
            # todo: pimg, pca
        }

        dgl.save_graphs('/data/zilu/files/cntGraph_Initial', graphs, labels=data)

if __name__ == '__main__':
    path = '/data/zilu/cnt/'

    with open('/data/zilu/files/cntmapping') as f:
        mapping = json.loads(f.read())

    data = pd.read_csv('data.csv', index_col=0, usecols=list(range(1,10)) + [14, 15])

    cnts = np.load('/data/zilu/files/ord.npy')

    graphdataset = dglGraph(path, mapping, data)

    graphdataset.get_write_dataset(cnts)
