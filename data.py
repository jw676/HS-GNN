'''
constructing different level of graphs
'''
import dgl
import numpy as np
import networkx as nx
import os
import json
import scipy.sparse as sp
import multiprocessing as mp
import random
import torch
from dgl.data import DGLDataset
# utilities

def read_edgelist_cnt(filename):
    bond = 0
    first = 0
    edgelist = []
    with open(filename+'/exp.data') as f:
        for line in f:
            rowlist = line.strip().split()
            if bond==0:
                if len(rowlist) !=0 and rowlist[0] == 'Bonds':
                    bond = 1
            elif len(rowlist)==0: # bond==1
                if first == 0:
                    first = 1
                else:
                    break
            else:
                edgelist.append((eval(rowlist[2]), eval(rowlist[3])))
    return edgelist

def read_coord_cnt(filename):
    atom = 0
    first = 0
    atoms = {}
    with open(filename+'/exp.data') as f:
        for line in f:
            rowlist = line.strip().split()
            if atom==0:
                if len(rowlist) !=0 and rowlist[0] == 'Atoms':
                    atom = 1
            elif len(rowlist)==0: # atom==1
                if first == 0:
                    first = 1
                else:
                    break
            else:
                atoms[eval(rowlist[0])] = np.array((eval(rowlist[4]), eval(rowlist[5]), eval(rowlist[6])))
    return atoms

def read_dihedrals_cnt(filename):
    dihedral = 0
    first = 0
    dihedrals = []
    with open(filename+'/exp.data') as f:
        for line in f:
            rowlist = line.strip().split()
            if dihedral==0:
                if len(rowlist) !=0 and rowlist[0] == 'Dihedrals':
                    dihedral = 1
            elif len(rowlist)==0: # atom==1
                if first == 0:
                    first = 1
                else:
                    break
            else:
                dihedrals.append([eval(rowlist[2]), eval(rowlist[5])])
    return dihedrals

def calculate_dE(graph, coords, mean=True):

    dE = [np.linalg.norm(coords[u] - coords[v]) for u, v in graph.edges()]

    if mean:  return np.mean(dE)
    else:  return np.array(dE)

def calculate_dR(graph, mean=True):

    # resistance distance must be calculated from a connected graph

    if nx.is_connected(graph):

        tinv = np.linalg.inv(nx.laplacian_matrix(graph).todense() + 1 / len(graph))  # tao inv matrix
        dR = [tinv[u, u] + tinv[v, v] - 2 * tinv[u, v] for u, v in graph.edges()]


    else:
        dR = []
        for nodes in nx.connected_components(graph):
            subg = nx.convert_node_labels_to_integers(nx.subgraph(graph, nodes))
            tinv = np.linalg.inv(nx.laplacian_matrix(subg).todense() + 1 / len(subg))  # tao inv matrix
            dR += [tinv[u, u] + tinv[v, v] - 2 * tinv[u, v] for u, v in subg.edges()]

    if mean:  return np.mean(dR)
    else:  return np.array(dR)

def read_write_bdc(filename):

    if os.path.exists(os.path.join(filename, 'eB.npy')) and os.path.exists(os.path.join(filename, 'eD.npy')):
        return

    edgelist = read_edgelist_cnt(filename)
    coords = read_coord_cnt(filename)

    graph = nx.from_edgelist(edgelist)

    if len(coords) != len(graph):  # in case there is isolated nodes
        print('Isolated atom in graph '+filename)


    for node in graph.nodes():
        graph.nodes()[node]['coord'] = coords[node]
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, label_attribute='old')
    idmapping = {node[1]['old']: node[0] for node in graph.nodes(data=True)}

    with open(filename + 'idmapping', 'w') as f:
        f.write(json.dumps(idmapping))

    eD = read_dihedrals_cnt(filename)
    eD_new = np.array([(idmapping[edge[0]], idmapping[edge[1]]) for edge in eD])

    np.save(os.path.join(filename, 'eB.npy'), np.array(graph.edges()))
    np.save(os.path.join(filename, 'eD.npy'), eD_new)

    coords = np.array([graph.nodes()[node]['coord'] for node in range(len(graph))])
    np.save(os.path.join(filename, 'coord.npy'), coords)

    return graph

def write_eE(filename, cutoff, cutoff_v2=None):

    edgelist = np.load(os.path.join(filename, 'eB.npy'))
    graph =  nx.from_edgelist(edgelist)
    coords = np.load(os.path.join(filename, 'coord.npy'))
    eE = []
    eE_v2 = []
    num_nodes = len(graph)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dE = np.linalg.norm(coords[i]-coords[j])
            if  dE <= cutoff:  eE.append([i,j])
            if cutoff_v2 is not None:
                if dE <=cutoff_v2:  eE_v2.append([i,j])

    np.save(os.path.join(filename, 'eE.npy'), eE)
    if not cutoff_v2 is None:
        np.save(os.path.join(filename, 'eE_v2.npy'), eE_v2)

    #print(filename + '  finshed eE')


def write_eR(filename, cutoff, cutoff_v2=None):

    edgelist = np.load(os.path.join(filename, 'eB.npy'))
    graph = nx.from_edgelist(edgelist)
    eR = []
    eR_v2 = []

    for setnodes in nx.connected_components(graph):
        subg = nx.subgraph(graph, setnodes)

        subg = nx.convert_node_labels_to_integers(subg, first_label=0, label_attribute='old')
        idmapping = {node[0]: node[1]['old']  for node in subg.nodes(data=True)}

        num_nodes = len(subg)
        tinv = np.linalg.inv(nx.laplacian_matrix(subg).todense() + 1 / len(subg))  # tao inv matrix

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dR = tinv[i, i] + tinv[j, j] - 2 * tinv[i, j]
                if dR <= cutoff:
                    eR.append([idmapping[i], idmapping[j]])
                if not cutoff_v2 is None:
                    eR_v2.append([idmapping[i], idmapping[j]])

    np.save(os.path.join(filename, 'eR.npy'), eR)
    if not cutoff_v2 is None:
        np.save(os.path.join(filename, 'eR_v2.npy'), eR_v2)
    #print(filename + '  finshed eR')

def wrap_write_eE(args):
    write_eE(*args)

def wrap_write_eR(args):
    write_eR(*args)

class coarsen():
    def __init__(self,
                 deltas):
        '''

        :param deltas: list, [delta1, delta2]
        '''
        self.deltas = deltas

    def get_write_graphs(self,
                         filename):
        '''

        :param filename: cnt directory
        :return:
        '''
        edge = np.load(os.path.join(filename, 'eB.npy'))
        coords = np.load(os.path.join(filename, 'coord.npy'))
        graph = nx.from_edgelist(edge)

        graphs = [graph]
        gCoords = [coords]

        for i, delta in enumerate(self.deltas):
            layer = i + 2
            piNodes, gCoords_, graph_, adj_ = self.get_coarser_graph(graphs[-1], gCoords[-1], delta)
            graphs.append(graph_)
            gCoords.append(gCoords_)
            piNodes = {j: list(piNodes[j]) for j in range(len(piNodes))}
            with open(os.path.join(filename, 'cluster_L'+str(layer)), 'w') as f:
                f.write(json.dumps(piNodes))
            sp.save_npz(os.path.join(filename, 'adj_L'+str(layer) + '.npz'), adj_)
        print(filename)

    def wrap(self, args):
        self.get_write_graphs(*args)

    def get_cluster(self,
                    point,
                    coords,
                    delta):
        pCoord = coords[point]
        cluster = set()
        for node, coord in enumerate(coords):
            if np.linalg.norm(coord-pCoord) <= delta:
                cluster.add(node)

        return cluster, pCoord

    def get_coarser_graph(self,
                          graph,
                          coords,
                          delta):
        '''

        :param graph: input graph at lower layer
        :param coords: node coords
        :param delta: cut off threshold
        :return piNodes: coarser graph clusters
        :return gCoords: coarser graph super node coords, for L2 layer
        :return cgraph: coarser graph
        :return adj: coarser graph adj

        '''

        allnodes = set(graph.nodes())
        pPrime = set() # original nodes having traversed
        piNodes = [] # coarse graph nodes
        gCoords = []  # coordinates for coarse graph nodes, required for L2 layer
        while len(pPrime) != len(allnodes):
            sample = random.choice(list(allnodes.difference(pPrime)))
            cluster, coord = self.get_cluster(sample, coords, delta)
            pPrime = pPrime.union(cluster)
            piNodes.append(cluster)
            gCoords.append(coord)

        num_cg_nodes = len(piNodes)
        cgraph = nx.Graph()
        for i in range(num_cg_nodes): cgraph.add_node(i)
        cgraph.add_nodes_from(range(num_cg_nodes))

        for i in range(num_cg_nodes):
            for j in range(i+1, num_cg_nodes):
                if len(piNodes[i].intersection(piNodes[j])) > 0:
                    cgraph.add_edge(i, j)

        adj = nx.adjacency_matrix(cgraph)

        return piNodes, gCoords, cgraph, adj

class CNTDataset(DGLDataset):
    def __init__(self,
                 file_path=None,
                 norm=False,
                 device=None):
        self.file_path = file_path
        self.norm = norm
        self.device = device

        super(CNTDataset, self).__init__(name='HSGNN', verbose=True)


    def has_cache(self):

        return os.path.exists(self.file_path)

    def process(self):

        # todo generate the graph
        pass

    def save(self):

        # todo save processed dataset
        pass

    def load(self):

        self.graphs, labels = dgl.load_graphs(self.file_path)

        self.targets = labels['labels'] # targets
        self.features = labels['feats'] # features


        if 'preds' in labels:
            self.preds = labels['preds']
        else:
            self.preds = None

        if self.norm:  self.normalize()

        if self.device is not None:
            self.graphs = [graph.to(self.device) for graph in self.graphs]
            self.features = self.features.to(self.device)
            self.targets = self.targets.to(self.device)
            if self.preds is not None: self.preds = self.preds.to(self.device)

        #print(self.graphs[0].nodes['C2'].data['pimg'].shape)

    def normalize(self):

        maxi, mini = torch.max(self.features, dim=0)[0], torch.min(self.features, dim=0)[0]
        self.features = (self.features - mini + 1e-2) / (maxi - mini + 1e-2)



        if self.preds is not None:
            maxi, mini = torch.max(self.preds, dim=0)[0], torch.min(self.preds, dim=0)[0]
            self.preds = (self.preds - mini + 1e-2) / (maxi - mini + 1e-2)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):

        '''
        if self.preds is None:
            return self.graphs[item], self.features[item, :], self.targets[item, :]
        else:
            return self.graphs[item], self.preds[item, :], self.targets[item, :]
        '''
        return self.graphs[item], self.features[item, :], self.targets[item, :]



if  __name__=='__main__':

    savedir = '/data/zilu/cnt/'
    allcnts = np.load('/data/zilu/files/ord.npy')
    with open('/data/zilu/files/cntmapping') as f:
        allfiles = json.loads(f.read())
    cntnames = [allfiles[str(i)] for i in allcnts]

    graphs = []
    for cnt in cntnames:
        graphs.append(read_write_bdc(os.path.join(savedir, cnt)))
    
    graphs = []
    coords = []
    for cnt in cntnames:
        graphs.append(nx.from_edgelist(np.load(os.path.join(savedir+cnt, 'eB.npy'))))
        coords.append(np.load(os.path.join(savedir+cnt, 'coord.npy')))

    #sampled_graphs = random.sample(graphs, k=100)

    sampled_indices = random.sample(list(range(len(graphs))), k=100)

    dEs = [calculate_dE(graph, coord, mean=False) for graph, coord in zip(
        [graphs[i] for i in sampled_indices], [coords[i] for i in sampled_indices])]
    dRs = [calculate_dR(graph, mean=False) for graph in [graphs[i] for i in sampled_indices]]

    avg_dE_mean = np.mean([np.mean(dE) for dE in dEs])
    avg_dR_mean = np.mean([np.mean(dR) for dR in dRs])

    avg_dE_all = np.mean(np.concatenate(dEs))
    avg_dR_all = np.mean(np.concatenate(dRs))

    print('Avg of Mean, dE: {:.4f}, dR: {:.4f}'.format(avg_dE_mean, avg_dR_mean))
    print('Avg of All,  dE: {:.4f}, dR: {:.4f}'.format(avg_dE_all, avg_dR_all))

    '''
    avg_dR_mean, avg_dR_all = 0.7261, 0.6760
    avg_dE_mean, avg_dE_all = 2.2985, 2.3219
    '''

    pool = mp.Pool(10)

    pool.map(wrap_write_eE, [(os.path.join(savedir, cnt), 6 * avg_dE_mean, 6 * avg_dE_all) for cnt in cntnames])
    pool.map(wrap_write_eR, [(os.path.join(savedir, cnt), 10 * avg_dR_mean, 10 * avg_dR_all) for cnt in cntnames])

    delta1 = 3 * avg_dE_mean
    delta2 = 6 * avg_dE_mean

    coarseGraph = coarsen([delta1, delta2])

    pool.map(coarseGraph.wrap, [(os.path.join(savedir, cnt), ) for cnt in cntnames])



