from dimenet import DimeNet
import torch
import torch.nn.functional as F
import random
import dgl
import numpy as np
from dgl.data.utils import Subset
from data import CNTDataset
from torch.utils.data import DataLoader

def split_dataset(dataset, num_train, num_test, shuffle=True, random_state=2):
    from itertools import accumulate
    num_data = len(dataset)
    assert num_train + num_test <= num_data
    lengths = [num_train, num_test]
    if shuffle:
        indices = np.random.RandomState(seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)

    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(accumulate(lengths), lengths)]

def collate_fn(batch):
    graphs, line_graphs, targets, features = map(list, zip(*batch))
    g, l_g = dgl.batch(graphs), dgl.batch(line_graphs)
    targets = torch.stack(targets, dim=0)
    features = torch.stack(features, dim=0)
    return g, l_g, targets, features

def evaluation(predictions, targets):
    mse = F.mse_loss(predictions, targets) # [N, 2]

def loss_fn(logits, targets):
    mse = F.mse_loss(logits, targets)
    return torch.sqrt(mse / torch.sum(targets ** 2))  # prmse


if __name__=="__main__":

    device = 'cuda:4'
    
    dataset = CNTDataset('/data/zilu/hsgnn/models/', device=device)
    print('Dataset loaded.')

    batch_size = 32

    train_ratio = 0.9
    num_train =int(train_ratio * len(dataset))
    num_test = len(dataset) - num_train

    train_data, test_data = split_dataset(dataset, num_train, num_test, shuffle=True, random_state=2)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             collate_fn=collate_fn,
                             shuffle=False)


    model = DimeNet(
        emb_size=32,
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        cutoff=1.6,
        envelop_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_dense_output=3,
        num_targets=1,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for e in range(100):
        train_loss = 0
        for g, l_g, targets, feats in train_loader:
            optimizer.zero_grad()
            print(e)
            logits = model(g, l_g)
            loss = loss_fn(logits.squeeze(), targets[:, 0])
            loss.backward()
            optimizer.step()
            train_loss += float(loss)

        print(train_loss/len(train_loader))


    model.eval()
    predictions = []
    labels = []
    for g, l_g, targets, feats in test_loader:
        preds = model(g, l_g)
        predictions.append(preds)
        targets = torch.cat([target.unsqueeze(0) for target in targets], dim=0)
        labels.append(targets)

    preds = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)[:, 0]
    print('test')
    print(loss_fn(preds.squeeze(), labels))
