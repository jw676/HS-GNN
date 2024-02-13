import argparse
import os.path
import random

import torch
from utilities import *
from utilities import _collate_fn
from sklearn.model_selection import train_test_split
from train import train, train_linear
from data import CNTDataset
from torch.utils.data import DataLoader


if __name__=="__main__":

    paser = argparse.ArgumentParser(description="HS-GNN")
    paser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    paser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    paser.add_argument("--device", type=str, default="6", help="Torch device")
    paser.add_argument("--epoch", type=int, default=40, help="Training epoch")
    paser.add_argument("--bs", type=int, default=32, help="Batch size")
    paser.add_argument("--p", type=int, default=3, help="Number of GIN convs in L1 layer")
    paser.add_argument("--q", type=int, default=3, help="Number of GAT convs in L1 layer")
    paser.add_argument("--concatbt", type=bool, default=True, help="Whether to concatenate L1 encoder (bottom) hidden representations")
    paser.add_argument("--num_head_bt", type=int, default=8, help="Number of GAT attention heads in L1 layer encoder")
    paser.add_argument("--num_inp", type=int, default=4, help="Number of input feature dimensions (atom feats) coord + degree")
    paser.add_argument("--num_hid", type=int, default=32, help="Number of hidden representation dimensions")
    paser.add_argument("--num_mlp_gin", type=int, default=3, help="Number of mlp layers in GIN, bottom layer")
    paser.add_argument("--init_eps", type=float, default=0.1, help="")
    paser.add_argument("--learn_eps", type=bool, default=True, help="")
    paser.add_argument("--gin_agg", type=str, default='mean', help="")
    paser.add_argument("--out_bt", type=str, default='mean', help="")
    paser.add_argument("--out_att", type=str, default='mean', help="")
    paser.add_argument("--pca_dim", type=int, default=9, help="")
    paser.add_argument("--pimg_dim", type=int, default=0, help="")
    paser.add_argument("--num_att_convs", type=int, default=3, help="")
    paser.add_argument("--num_heads_att", type=int, default=8, help="")
    paser.add_argument("--num_mlp_att", type=int, default=3, help="")
    paser.add_argument("--glb_dim", type=int, default=9, help="")
    paser.add_argument("--loss", type=str, default='mse', help="Loss type")
    paser.add_argument("--final_readout", type=str, default='mean', help="Can be ‘sum’, ‘max’, ‘min’, ‘mean’.")
    paser.add_argument("--predict", type=str, default='m', help="Can be ‘m’ (modulus), ‘s’ (strength), other unsupported yet.")
    paser.add_argument("--model", type=str, default='hsgnn', help="hsgnn, lr")
    paser.add_argument("--num_exp", type=int, default=5, help="number of exps")


    args = paser.parse_args()

    if args.device=="cpu":
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cuda:"+ args.device)

    filepath = '/data/zilu/files/cntGraph_basic'


    dataset = CNTDataset(file_path=filepath)

    # split the dataset
    training_ratio = 0.9
    num_train = int(training_ratio * len(dataset))
    num_test = len(dataset) - num_train

    test_performance = []

    if args.model == 'hsgnn':

        seeds = [random.randint(0, int(1e5)) for _ in range(args.num_exp)]

        for i, seed in enumerate(seeds):

            print('Exp {:2d}/{:2d}'.format(i+1, args.num_exp))

            train_data, test_data = split_data(dataset, num_train, num_test, shuffle=True, random_state=seed)

            train_loader = DataLoader(train_data,
                                      batch_size=args.bs,
                                      shuffle=True,
                                      collate_fn=_collate_fn)

            test_loader = DataLoader(test_data,
                                     batch_size=args.bs,
                                     shuffle=True,
                                     collate_fn=_collate_fn)

            model = train(args, train_loader)

            test_loss = evaluate(model, test_loader, args.device)

            test_performance.append(test_loss)

            del model
            torch.cuda.empty_cache()

        test_performance = np.array(test_performance)
        np.save('/data/zilu/hsgnn/hsgnntestm.npy', test_performance)


    elif args.model == 'lr':

        seeds = [random.randint(0, int(1e5)) for _ in range(args.num_exp)]

        for i, seed in enumerate(seeds):
            print('Exp {:2d}/{:2d}'.format(i + 1, args.num_exp))

            train_data, test_data = split_data(dataset, num_train, num_test, shuffle=True, random_state=seed)

            train_loader = DataLoader(train_data,
                                      batch_size=args.bs,
                                      shuffle=True,
                                      collate_fn=_collate_fn)

            test_loader = DataLoader(test_data,
                                     batch_size=args.bs,
                                     shuffle=True,
                                     collate_fn=_collate_fn)

            model = train_linear(args, train_loader)

            test_loss = evaluate_lr(model, test_loader, args.device)

            test_performance.append(test_loss)

        test_performance = np.array(test_performance)
        np.save('/data/zilu/hsgnn/lrtestm.npy', test_performance)

    else:
        raise KeyError('Unsupported Model type')


