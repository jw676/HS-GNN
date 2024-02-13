import numpy as np
import json
import os
import dgl
import torch
import pandas as pd
import time
from dgl.data import DGLDataset

class DNGraph():
    # for DimeNet input
    def __init__(self, datadir, cnts, threshold, glbfeats, savedir):
        self.datadir = datadir
        self.cnts = cnts
        self.cutoff = threshold
        self.glbfeats = glbfeats
        self.savedir = savedir

    def init_edges(self, edges):
        src, dst = edges.src['R'], edges.dst['R']
        edge_orien = src - dst
        edge_dist = torch.norm(src-dst, dim=-1)
        return {'d': edge_dist, 'o': edge_orien}

    def oneGraph(self, cnt):
        pwdist = np.load(os.path.join(self.datadir + cnt, 'pwdist.npy'))
        if pwdist.shape[0]==pwdist.shape[1]:
            src, dst = np.where(np.triu(pwdist <= self.cutoff, k=1))
        else:
            edges = pwdist[pwdist[:, -1] <= self.cutoff][:, :2]
            src, dst = edges[:, 0], edges[:, 1]

        graph = dgl.graph((src, dst))

        coords = torch.FloatTensor(np.load(os.path.join(self.datadir+cnt, 'coord.npy')))

        graph.ndata['R'] = coords

        graph.apply_edges(self.init_edges)

        graph = dgl.add_reverse_edges(graph, copy_edata=True)

        degrees = graph.in_degrees()
        graph.ndata['Z'] = degrees - min(degrees)

        line_graph = dgl.line_graph(graph, backtracking=False)

        return graph, line_graph

    def bondGraph(self, cnt):
        bonds = np.load(os.path.join(self.datadir+cnt, 'eB.npy'))

        graph = dgl.graph((bonds[:, 0], bonds[:, 1]))

        coords = torch.FloatTensor(np.load(os.path.join(self.datadir+cnt, 'coord.npy')))

        graph.ndata['R'] = coords

        graph.apply_edges(self.init_edges)

        graph = dgl.add_reverse_edges(graph, copy_edata=True)

        degrees = graph.in_degrees()
        graph.ndata['Z'] = degrees - min(degrees)

        line_graph = dgl.line_graph(graph, backtracking=False)

        return graph, line_graph

    def bondGraphcut(self, cnt):
        bonds = np.load(os.path.join(self.datadir+cnt, 'eB.npy'))

        coords = np.load(os.path.join(self.datadir+cnt, 'coord.npy'))

        bonds = bonds[np.linalg.norm(coords[bonds[:, 0]] - coords[bonds[:, 1]], axis=1) < self.cutoff]

        graph = dgl.graph((bonds[:, 0], bonds[:, 1]))

        graph.ndata['R'] = torch.FloatTensor(coords)

        graph.apply_edges(self.init_edges)

        graph = dgl.add_reverse_edges(graph, copy_edata=True)

        degrees = graph.in_degrees()
        graph.ndata['Z'] = degrees - min(degrees)

        line_graph = dgl.line_graph(graph, backtracking=False)

        return graph, line_graph

    def getGraphs(self, verbose=True):
        self.graphs, self.line_graphs = [], []
        features, targets = [], []
        start = time.time()

        for i, cnt in enumerate(self.cnts):
            if verbose and i % 50 == 0:
                cost = time.time() - start
                print(cnt, cost/60)

            #g, lg = self.oneGraph(cnt)
            g, lg = self.bondGraph(cnt)
            self.graphs.append(g)
            self.line_graphs.append(lg)

            glbfeatures = self.glbfeats.loc[cnt].to_numpy()
            features.append(glbfeatures[:9])
            targets.append(glbfeatures[9:])

        features = torch.FloatTensor(np.array(features))
        targets = torch.FloatTensor(np.array(targets))
        self.labels = {'features': features, 'targets': targets}

        return self.graphs, self.line_graphs, self.labels

    def save(self):
        dgl.save_graphs(self.savedir+'DimeGraph_c', self.graphs, labels=self.labels)
        dgl.save_graphs(self.savedir+'LineGraph_c',self.line_graphs)


class CNTDataset(DGLDataset):
    def __init__(self,
                 raw_dir=None,
                 device='cpu'):
        self.dimepath = os.path.join(raw_dir, 'DimeGraph_c')
        self.linepath = os.path.join(raw_dir, 'LineGraph_c')
        #self.dimepath = os.path.join(raw_dir, 'dg')
        #self.linepath = os.path.join(raw_dir, 'lg')
        self.device = device

        super(CNTDataset, self).__init__(name='DimeNet')



    def has_cache(self):
        '''
        step 1
        if Ture goto step 5
        else goto step 2 then step 3 
        '''
        #print(self._force_reload)
        return os.path.exists(self.dimepath) and os.path.exists(self.linepath)

    def process(self):
        '''
        step 3
        '''
        print(1)
        datadir = '/data/zilu/cnt/'

        cntindex = np.load('/data/zilu/files/ord.npy')
        with open('/data/zilu/files/cntmapping') as f:
            mapping = json.loads(f.read())
        cnts = [mapping[str(i)] for i in cntindex]

        avgDist = 1.42
        threshold = avgDist

        glbfeats = pd.read_csv('/data/zilu/hsgnn/data.csv', index_col=0, usecols=list(range(1, 11)) + [14, 15])


        self.toGraph = DNGraph(datadir, cnts, threshold, glbfeats, self.raw_dir)

        self.graphs, self.line_graphs, labels = self.toGraph.getGraphs(verbose=True)
        self.targets = labels['targets']
        self.features = labels['features']

    def save(self):
        '''
        step 4
        '''
        self.toGraph.save()

    def load(self):
        '''
        step 5
        '''
        self.graphs, labels = dgl.load_graphs(self.dimepath)
        self.line_graphs, _ = dgl.load_graphs(self.linepath)

        self.graphs = [graph.to(self.device) for graph in self.graphs]
        self.line_graphs = [graph.to(self.device) for graph in self.line_graphs]

        self.targets = labels['targets'].to(self.device)
        self.features = labels['features'].to(self.device)



    def __getitem__(self, idx):
        return self.graphs[idx], self.line_graphs[idx], self.targets[idx], self.features[idx]

    def __len__(self):
        return len(self.graphs)

if __name__ == '__main__':
    datadir = '/data/zilu/cnt/'

    cntindex = np.load('/data/zilu/files/ord.npy')
    with open('/data/zilu/files/cntmapping') as f:
        mapping = json.loads(f.read())
    cnts = [mapping[str(i)] for i in cntindex]

    avgDist = 1.42
    threshold = 3 * avgDist

    glbfeats = pd.read_csv('../data.csv', index_col=0, usecols=list(range(1,11)) + [14, 15])

    savedir = ''

    toGraph = DNGraph(datadir, cnts, threshold, glbfeats, savedir)

    toGraph.getGraphs()

    toGraph.save()

