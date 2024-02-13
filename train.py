import os.path
import random
import argparse
import dgl
import torch
from utilities import *
import copy
import datetime
from data import CNTDataset
from torch.utils.data import DataLoader
from hsgnn_models import BTEC, GAT, GIN, GAT_N, H2, H2_, H2_V, H2_N, H2_GAT, H2_CAT, H2_EdgeAtt, H3_, H3_GLB, H3_PI, \
    H2_GLB, MLP, H2_RNI, BEP, H2_PE, H2_PICA, H3_PCA

def early_stop(best, losses):
    best = losses[-1]
    deviation = abs(np.array(losses) - best) / best
    if np.mean(deviation) < 0.02:
        return True
    return False

def train_model(args, train_loader, valid_loader, test_loader, large_loader, valid, milestones=None, fold=1):
    #'''
    model = H3_(
        bottom_edge_types=bottom_edge_types,
        node_types='uniform',
        in_dim=args.num_inp,
        hid_dim=args.num_hid,
        out_dim=1,
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
    '''
    model = BTEC(
        bottom_edge_types=bottom_edge_types,
        node_types='uniform',
        in_dim=args.num_inp,
        hid_dim=args.num_hid,
        out_dim=1,
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
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.3)
    lossfunc = loss_rmrse

    ares, rmrses = [], []

    best_are, best_rmrse = 1e10, 1e10
    best_are_lrg, best_rmrse_lrg = 1e10, 1e10
    best_are_val, best_rmrse_val = 1e10, 1e10
    best_epoch = None

    for e in range(args.epoch):
        model.train()

        for graph, feats, targets in train_loader:
            optimizer.zero_grad()
            if not args.add_glb:
                logits = model(graph.to(args.device))
            else:
                logits = model(graph, feats)
            loss = lossfunc(logits, targets.to(args.device))
            loss.backward()
            optimizer.step()
        if milestones is not None:
            scheduler.step()

        if e % valid == 0 or e == args.epoch-1:
            model.eval()
            #print('Epoch '+ str(e))
            #evaluate(model, train_loader, args, echo=True, test=False)
            vare, vmrse, vrmrse = evaluate(model, valid_loader, args, echo=False)
            #evaluate(model, test_loader, args, echo=True, test=True)
            #evaluate(model, large_loader, args, echo=True, test=False, large=True)

            ares.append(vare)
            rmrses.append(vrmrse)

            if vare < best_are_val:
                best_are_val = vare
                best_rmrse_val = vrmrse
                best_epoch = e
                bestmodel = copy.deepcopy(model)

            if e == 0: continue

            if e > 100 and not (best_are_val == np.array(ares[-5:])).any() and \
                    early_stop(best_are_val, ares[-10:]) and early_stop(best_rmrse_val, rmrses[-10:]):
                print('Early stop')
                break

            if vare < min(ares[:-1]):
                tare, tmrse, trmrse = evaluate(model, test_loader, args, echo=False)
                best_are = tare
                best_rmrse = trmrse

                if args.test_lrg:
                    lare, lmrse, lrmrse = evaluate(model, large_loader, args, echo=False)
                    best_are_lrg = lare
                    best_rmrse_lrg = lrmrse
    print('Best Epoch [{:3d} / {:3d}]'.format(best_epoch, e))
    print('Valid ARE {:.2f} | Valid RMRSE {:.2f}'.format(best_are_val * 100, best_rmrse_val * 100))
    print('Test ARE {:.2f} | Test RMRSE {:.2f}'.format(best_are * 100, best_rmrse * 100))
    if args.test_lrg:
        print('Large ARE {:.2f} | Large RMRSE {:.2f}'.format(best_are_lrg * 100, best_rmrse_lrg * 100))

    torch.save(bestmodel.state_dict(), 'trainedmodel/hsgnn3_wo_'+args.predict+str(fold))
    del model, bestmodel
    torch.cuda.empty_cache()

    return best_are, best_rmrse, best_are_lrg, best_rmrse_lrg

if __name__=="__main__":


    paser = argparse.ArgumentParser(description="HS-GNN")
    paser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    paser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    paser.add_argument("--device", type=str, default="1", help="Torch device")
    paser.add_argument("--epoch", type=int, default=300, help="Training epoch")
    paser.add_argument("--bs", type=int, default=512, help="Batch size")
    paser.add_argument("--p", type=int, default=2, help="Number of GIN convs in L1 layer")
    paser.add_argument("--q", type=int, default=2, help="Number of GAT convs in L1 layer")
    paser.add_argument("--concatbt", type=bool, default=True, help="Whether to concatenate L1 encoder (bottom) hidden representations")
    paser.add_argument("--num_head_bt", type=int, default=4, help="Number of GAT attention heads in L1 layer encoder")
    paser.add_argument("--num_inp", type=int, default=4, help="Number of input feature dimensions (atom feats) coord + degree")
    paser.add_argument("--num_hid", type=int, default=32, help="Number of hidden representation dimensions")
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
    paser.add_argument("--test_lrg", type=bool, default=True, help="")
    paser.add_argument("--loss", type=str, default='mse', help="Loss type")
    paser.add_argument("--final_readout", type=str, default='mean', help="Can be ‘sum’, ‘max’, ‘min’, ‘mean’.")
    paser.add_argument("--predict", type=str, default='m', help="Can be ‘m’ (modulus), ‘s’ (strength), other unsupported yet.")
    paser.add_argument("--model", type=str, default='hsgnn', help="hsgnn, lr")
    paser.add_argument("--att_type", type=str, default='pimg', help="Can be pca, pimg. / pcavec, pcaval, pcarat")
    paser.add_argument("--objective", type=str, default='mrse', help="are, mrse, rmrse")
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
    print('Predicting '+ args.predict)
    if args.num_inp == 4:
        print('without global')
    elif args.num_inp == 12:
        print('with global')

    print('device: ' + args.device)
    #milestones = [20, 100, 200, 400, 600]
    milestones = [20, 60, 140, 240, 400]

    bottom_edge_types = ['B', 'D', 'E']

    filepath = '/data/zilu/files/cntGraphE_3'

    if args.predict == 's':
        _collate_fn = collate_fn_s
    elif args.predict == 'm':
        _collate_fn = collate_fn_m
    else:
        raise KeyError('only predicting s or m')

    if args.device=="cpu":
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cuda:"+ args.device)
    dataset = CNTDataset(file_path=filepath, norm=args.normalize, device=args.device)

    index = np.load('newidx.npy')

    if args.test_lrg:
        lrgidx = np.load('/data/zilu/hsgnn/lrg_test_idx.npy')
        lrg_dataset = Subset(dataset, list(lrgidx))
        lrg_dataloader = DataLoader(
            lrg_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=_collate_fn,
            drop_last=False
        )

    num_test = int(0.1 * len(index))

    num_valid = num_test

    num_exps = 5

    results = []

    for cf in range(10):
        test_idx = index[cf* num_test : (cf+1) * num_test]
        other_idx = copy.deepcopy(np.append(index[(cf+1)*num_test:], index[:cf*num_test]))
        val_idx = other_idx[:num_valid]
        train_idx = other_idx[num_valid:]

        train_data = Subset(dataset, list(train_idx))
        valid_data = Subset(dataset, list(val_idx))
        test_data = Subset(dataset, list(test_idx))

        test_loader = DataLoader(test_data,
                                  batch_size=args.bs,
                                  shuffle=False,
                                  collate_fn=_collate_fn,
                                  drop_last=False)

        train_loader = DataLoader(train_data,
                                  batch_size=args.bs,
                                  shuffle=True,
                                  collate_fn=_collate_fn,
                                  drop_last=True)

        valid_loader = DataLoader(valid_data,
                                  batch_size=args.bs,
                                  shuffle=False,
                                  collate_fn=_collate_fn,
                                  drop_last=False)
        #cvresult = []

        cvresult = train_model(args, train_loader, valid_loader, test_loader, lrg_dataloader, 5, milestones=None, fold=cf)

        results.append(cvresult)
        #print(datetime.datetime.now())

        print('Fold '+str(cf))
        print(cvresult)

        print('**************************')

        results.append(cvresult)

    results = np.array(results)
    print('Final Result: ', end='    ')
    print(np.mean(results, axis=0))
    np.save(args.predict+'result/hsgnn3_wo.npy', results)
    print()
