import gudhi as gd
from gudhi.representations.vector_methods import PersistenceImage as PI
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.linalg import eigh
import random
import os
import time
import json


class PH():
    def __init__(self, filename, t=1, filt_func='hks', resolution=[8,8]):
        if filt_func not in ('hks', 'degree'):
            raise KeyError('Not supported filtration function.')
        self.t = t
        self.filt_func = filt_func

        self.PI = PI(resolution=resolution)

        self.persimg = []
        self.epds = {}

        self.filename = filename

    def dg2pi(self, diagrams):

        pi = self.PI.transform(diagrams)
        pi = np.expand_dims(pi, axis=0)
        self.persimg.append(pi)

    def get_pd_pi(self):

        graph = nx.from_edgelist(np.load(os.path.join(self.filename, 'eB.npy')))
        with open(os.path.join(self.filename, 'cluster_L2')) as f:
            cluster = json.loads(f.read())
        superNodes = list(cluster.keys())
        superNodes.sort()

        for i in superNodes:
            sg = nx.subgraph(graph, cluster[i]).copy()
            epds = self.graph2epds(sg)
            self.dg2pi(epds)
            self.epds[i] = [pd.tolist() for pd in epds]

        self.persimg = np.concatenate(self.persimg, axis=0)

    def write_pd_pi(self):

        with open(os.path.join(self.filename, 'epds_L2'), 'w') as f:
            f.write(json.dumps(self.epds))

        np.save(os.path.join(self.filename, 'pi_L2'), self.persimg)


    def compute_hks(self, eigenvectors, eigenvalues, time):
        return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvalues))).sum(axis=1)


    def graph2epds(self, graph):
        adj = nx.adjacency_matrix(graph).todense()

        if self.filt_func == 'hks':
            L = csgraph.laplacian(adj, normed=True)
            egvals, egvecs = eigh(L)
            filt_val = self.compute_hks(egvecs, egvals, self.t)
        elif self.filt_func == 'degree':
            filt_val = adj.sum(axis=1)

        return self.graph_ex_ph(adj, filt_val)

    def graph_ex_ph(self, A, filtration_val):
        num_vertices = A.shape[0]
        (xs, ys) = np.where(np.triu(A))
        st = gd.SimplexTree()
        for i in range(num_vertices):
            st.insert([i], filtration=-1e10)
        for idx, x in enumerate(xs):
            st.insert([x, ys[idx]], filtration=-1e10)
        for i in range(num_vertices):
            st.assign_filtration([i], filtration_val[i])
        st.make_filtration_non_decreasing()
        st.extend_filtration()
        LD = st.extended_persistence()

        dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = self.dgm2np(LD[0], 0), self.dgm2np(LD[1], 1), self.dgm2np(LD[2], 0), self.dgm2np(LD[3], 1)

        return dgmOrd0, dgmExt0, dgmRel1, dgmExt1

    def dgm2np(self, dgm, dimension):
        return np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgm if p[0] == dimension]
                         ) if len(dgm) else np.empty([0, 2])



if __name__ == '__main__':

    cntidx = np.load('/data/zilu/files/ord.npy')
    with open('/data/zilu/files/cntmapping') as f:
        mapping = json.loads(f.read())

    for cnt in cntidx:
        filename = '/data/zilu/cnt/'+mapping[str(cnt)]
        ph = PH(filename)
        ph.get_pd_pi()
        ph.write_pd_pi()

