import copy
import os.path
import random
import argparse
import dgl
import torch
from utilities import *

from data import CNTDataset
from torch.utils.data import DataLoader
from hsgnn_models import BTEC, GAT, GIN, GAT_N, H2, H2_, H2_V, H2_N, H2_GAT, H2_CAT, H2_EdgeAtt, H3_, H3_GLB, H3_PI, \
    H2_GLB, MLP, H2_RNI, BEP, H2_PE, H2_PICA, H3_PCA
from contrast import SSL
import torch.nn as nn

if __name__=="__main__":


    paser = argparse.ArgumentParser(description="HS-GNN")
    paser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    paser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    paser.add_argument("--device", type=str, default="5", help="Torch device")
    paser.add_argument("--epoch", type=int, default=60, help="Training epoch")
    paser.add_argument("--bs", type=int, default=128, help="Batch size")
    paser.add_argument("--p", type=int, default=2, help="Number of GIN convs in L1 layer")
    paser.add_argument("--q", type=int, default=2, help="Number of GAT convs in L1 layer")
    paser.add_argument("--concatbt", type=bool, default=True, help="Whether to concatenate L1 encoder (bottom) hidden representations")
    paser.add_argument("--num_head_bt", type=int, default=4, help="Number of GAT attention heads in L1 layer encoder")
    paser.add_argument("--num_inp", type=int, default=4, help="Number of input feature dimensions (atom feats) coord + degree")
    paser.add_argument("--num_hid", type=int, default=64, help="Number of hidden representation dimensions")
    paser.add_argument("--num_out", type=int, default=128, help="Number of output representation dimensions")
    paser.add_argument("--num_mlp_gin", type=int, default=3, help="Number of mlp layers in GIN, bottom layer")
    paser.add_argument("--init_eps", type=float, default=0.1, help="")
    paser.add_argument("--learn_eps", type=bool, default=True, help="")
    paser.add_argument("--gin_agg", type=str, default='mean', help="")
    paser.add_argument("--out_bt", type=str, default='mean', help="")
    paser.add_argument("--out_att", type=str, default='mean', help="")
    paser.add_argument("--pca_dim", type=int, default=2, help="")
    paser.add_argument("--pimg_dim", type=int, default=64 * 2, help="") # it performs better when adding pimg
    paser.add_argument("--num_att_convs", type=int, default=1, help="")
    paser.add_argument("--num_heads_att", type=int, default=4, help="")
    paser.add_argument("--num_mlp_att", type=int, default=2, help="")
    paser.add_argument("--glb_dim", type=int, default=0, help="")
    paser.add_argument("--add_glb", type=bool, default=False, help="")
    paser.add_argument("--test_lrg", type=bool, default=False, help="")
    paser.add_argument("--loss", type=str, default='mse', help="Loss type")
    paser.add_argument("--final_readout", type=str, default='mean', help="Can be ‘sum’, ‘max’, ‘min’, ‘mean’.")
    paser.add_argument("--predict", type=str, default='s', help="Can be ‘m’ (modulus), ‘s’ (strength), other unsupported yet.")
    paser.add_argument("--model", type=str, default='hsgnn', help="hsgnn, lr")
    paser.add_argument("--att_type", type=str, default='pimg', help="Can be pca, pimg. / pcavec, pcaval, pcarat")
    paser.add_argument("--objective", type=str, default='are', help="are, mrse, rmrse")
    paser.add_argument("--save", type=bool, default=False)
    paser.add_argument("--num_exp", type=int, default=1, help="number of exps")
    paser.add_argument("--normalize", type=bool, default=True, help="whether to normalize global features")
    paser.add_argument("--equib", type=bool, default=True)


    args = paser.parse_args()
    #0 normalize, glb same upper bound 21 (27)
    #2 without glb, mean output ~32
    if args.att_type == 'pcaval':
        args.pca_dim = 2
    elif args.att_type == 'pcavec':
        args.pca_dim = 6
    elif args.att_type == 'pcarat':
        args.pca_dim = 2
    print(os.getpid())

    print('device: ' + args.device)
    #milestones = [20, 100, 200, 400, 600]


    bottom_edge_types = ['B', 'D']

    init_filepath = '/data/zilu/files/cntGraph_0'
    equib_filepath = '/data/zilu/files/cntGraphE_0'

    neg_pos_ratio = 5


    if args.device=="cpu":
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cuda:"+ args.device)

    init_dataset = dgl.load_graphs(init_filepath)[0]

    equib_dataset = dgl.load_graphs(equib_filepath)[0]


    for i in range(args.num_exp):

        print('---------------')
        print('exp '+ str(i))


        init_model = H3_(
            bottom_edge_types=bottom_edge_types,
            node_types='uniform',
            in_dim=args.num_inp,
            hid_dim=args.num_hid,
            out_dim=args.num_out,
            num_gin=args.p,
            num_gat=args.q,
            num_mlp_layers_gin=args.num_mlp_gin,
            num_heads_bt=args.num_head_bt,
            init_eps_gin=args.init_eps,
            learn_eps_gin=args.learn_eps,
            gin_agg=args.gin_agg,
            out_type_bt=args.out_bt,
            pca_dim=args.pca_dim,
            pi_dim=args.pimg_dim,
            num_mlp_layers_att=args.num_mlp_att,
            num_att_convs=args.num_att_convs,
            num_heads_att=args.num_heads_att,
            out_type_att=args.out_att,
            att_type=args.att_type,
            glb_dim=args.glb_dim,
            final_readout=args.final_readout,
            #atom_glb=True
        ).to(args.device)

        equib_model = H3_(
            bottom_edge_types=bottom_edge_types,
            node_types='uniform',
            in_dim=args.num_inp,
            hid_dim=args.num_hid,
            out_dim=args.num_out,
            num_gin=args.p,
            num_gat=args.q,
            num_mlp_layers_gin=args.num_mlp_gin,
            num_heads_bt=args.num_head_bt,
            init_eps_gin=args.init_eps,
            learn_eps_gin=args.learn_eps,
            gin_agg=args.gin_agg,
            out_type_bt=args.out_bt,
            pca_dim=args.pca_dim,
            pi_dim=args.pimg_dim,
            num_mlp_layers_att=args.num_mlp_att,
            num_att_convs=args.num_att_convs,
            num_heads_att=args.num_heads_att,
            out_type_att=args.out_att,
            att_type=args.att_type,
            glb_dim=args.glb_dim,
            final_readout=args.final_readout,
            # atom_glb=True
        ).to(args.device)

        model = SSL(init_model, equib_model, args.num_out).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lossfunc = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([neg_pos_ratio]).to(args.device))

        # generating pairs
        num_train = len(init_dataset)
        init_idx = list(range(num_train))
        equib_idx = copy.deepcopy(init_idx) * neg_pos_ratio


        for e in range(args.epoch):
            model.train()

            random.shuffle(equib_idx)

            epoch_labels = np.append(np.ones(num_train), np.zeros(num_train * neg_pos_ratio))
            epoch_idx = np.stack([np.array(init_idx + init_idx * neg_pos_ratio),
                                  np.array(init_idx + equib_idx),
                                  epoch_labels]).transpose().astype(np.int64)

            np.random.shuffle(epoch_idx)

            epoch_labels = epoch_idx[:, -1]
            epoch_idx = epoch_idx[:, :2]

            num_batch = len(epoch_idx) // args.bs

            logits = []

            for batch in range(num_batch):
                optimizer.zero_grad()
                batch_idx = epoch_idx[batch*args.bs : (batch+1)*args.bs]

                init_batch_graphs = dgl.batch([init_dataset[k] for k in batch_idx[:, 0]]).to(args.device)
                equib_batch_graphs = dgl.batch([equib_dataset[k] for k in batch_idx[:, 1]]).to(args.device)

                batch_labels = torch.FloatTensor(epoch_labels[batch*args.bs : (batch+1)*args.bs]).to(args.device)

                blogits = model(init_batch_graphs, equib_batch_graphs).squeeze()

                bloss = lossfunc(blogits, batch_labels)
                bloss.backward()

                logits.append(blogits.detach())

                optimizer.step()


            logits = torch.cat(logits)
            elabels = torch.FloatTensor(epoch_labels).to(args.device)

            print('Epoch ' + str(e), end=' | ')
            print('Train Loss ' + str(lossfunc(logits, elabels[:logits.size(0)]).item()))
            # we actually do not need test loss

        model.eval()
        init_model.eval()
        equib_model.eval()

        torch.save(init_model.state_dict(), 'trainedmodel/init_model_h3')
        torch.save(equib_model.state_dict(), 'trainedmodel/equib_model_h3')

        init_embedding = []
        equib_embedding = []

        for init_graph, equib_graph in zip(init_dataset, equib_dataset):

            init_hidden = init_model(init_graph.to(args.device)).cpu().detach().numpy()
            equib_hidden = equib_model(equib_graph.to(args.device)).cpu().detach().numpy()

            init_embedding.append(init_hidden)
            equib_embedding.append(equib_hidden)

        init_embedding = np.stack(init_embedding)
        equib_embedding = np.stack(equib_embedding)

        np.save('init_embedding_H3.npy', init_embedding)
        np.save('equib_embedding_H3.npy', equib_embedding)

        del model
        torch.cuda.empty_cache()
