import copy

import numpy as np
import torch.nn as nn
import torch
from nnutil import LinearRgressor, MLPG, MLPR

from hsgnn_models import MLP
import os

def normalize(data, colmin=None, colmax=None, returnminmax=False):
    #return np.sqrt(data)
    if colmax is None:
        colmax = np.max(data, axis=0)
    if colmin is None:
        colmin = np.min(data, axis=0)

    normed_feats = (data - colmin + 1e-5) / (colmax-colmin + 1e-5)

    if returnminmax:
        return normed_feats, colmin, colmax
    return normed_feats
    #return np.sqrt((data - colmin) / (colmax-colmin))

def mse(logits, labels):
    return torch.sum((logits - labels) ** 2)

def mae(logits, labels):
    return torch.sum(torch.abs(logits - labels))

def loss_are(logits, labels):
    # avg relative error
    relative_error = torch.abs(logits.squeeze() - labels) / labels
    return torch.mean(relative_error)

def loss_mrse(logits, labels):
    # mean relative square error
    rse = ((logits.squeeze() - labels) / labels)**2
    return torch.mean(rse)

def loss_rmrse(logits, labels):
    # root mean relative square error
    mrse = loss_mrse(logits, labels)
    return torch.sqrt(mrse)

def eval(model, data, labels, add_data=None, refine=False):
    model.eval()
    if refine:
        pred = model(data, add_data)
    else:
        pred = model(data)
    are = loss_are(pred, labels).item()
    mrse = loss_mrse(pred, labels).item()
    rmrse = loss_rmrse(pred, labels).item()

    return  are, mrse, rmrse

def early_stop(error, loss):
    mean = error
    deviation = np.abs(mean - np.array(loss)) / mean
    if np.mean(deviation) < 0.01:
        return True
    else: return False

def train(lr, feats, add_data, num_epochs, train_idx, test_idx, large_idx, milestones=None, test_lrg=False):

    num_inputs = feats.size(1)
    add_dim = add_data.size(1)
    #'''
    model = MLPG(
        in_dim=num_inputs,
        hid_dim=add_dim,
        num_layers=4,
        out_dim=1,
        add_layer=1
    ).to(device)
    '''

    model = MLPG(
        in_dim=num_inputs,
        hid_dim=add_dim,
        num_layers=5,
        out_dim=1,
        add_layer=2
    ).to(device)
    '''
    #model = nn.Linear(num_inputs, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.3)


    ares, rmrses, lares, lrmrses = [], [], [], []


    for e in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        if refined:
            logits = model(feats[train_idx], add_data[train_idx])
        else:
            logits = model(feats[train_idx])
        loss = loss_fn(logits, strength[train_idx])
        loss.backward()
        optimizer.step()
        if milestones is not None: scheduler.step()


        if e % valid == 0 or e==num_epochs-1:
            model.eval()

            are, mrse, rmrse = eval(model, feats[test_idx], strength[test_idx], add_data[test_idx], refined)

            ares.append(are)
            rmrses.append(rmrse)

            if test_lrg:
                lare, lmrse, lrmrse = eval(model, feats[large_idx], strength[large_idx], add_data[large_idx], refined)
                lares.append(lare)
                lrmrses.append(lrmrse)

    print('Best Test ARE {:.2f} | RMRSE {:.2f}'.format(are * 100, rmrse * 100))
    print('Best Large Test ARE {:.2f} | RMRSE {:.2f}'.format(lare * 100, lrmrse * 100))


    return ares, rmrses, lares, lrmrses


loss_fn = loss_rmrse
device = torch.device('cuda:2')

print(device)

print(os.getpid())

index = np.load('newidx.npy')

num_test = int(0.1 * len(index))

large_idx = torch.LongTensor(np.load('lrg_test_idx.npy')).to(device)

pt = 'm'
mapping = {'s':0, 'm':1}

print('predict '+pt)

feats = np.load('glb_feats.npy')
#hid = torch.FloatTensor(np.load('idx4/'+pt+'_hsgnn_hid.npy')).view(feats.shape[0], 1).to(device)
init_emb = torch.FloatTensor(np.load('init_embedding_H3.npy')).to(device).view(feats.shape[0], -1)
equib_emb = torch.FloatTensor(np.load('equib_embedding_H3.npy')).to(device).view(feats.shape[0], -1)

strength = torch.FloatTensor(np.load('all_targets.npy')[:, mapping[pt]]).to(device)

add_data = torch.cat([init_emb, equib_emb], dim=-1).to(device)

add_dim = add_data.size(1)

refined = True

only_hsgnn = False

linear = False

nontube = False

valid = 5

milestones = [500, 1000, 1500]

if only_hsgnn: feats = add_data

if nontube:
    ntidx = [1, 2, 3, 4, 6]
    feats = feats[:, ntidx]
    print(feats.shape)

num_inputs = feats.shape[1]

results = []

for cf in range(10):
    test_idx = index[cf*num_test : (cf+1)*num_test]
    train_idx = copy.deepcopy(np.append(index[(cf+1)*num_test:], index[:cf*num_test]))

    cfresults = []

    for i in range(5):
        print('Exp ', i)

        if only_hsgnn:
            feats_input = feats
        else:
            _, colmin, colmax = normalize(feats[train_idx], returnminmax=True)

            feats_input = torch.FloatTensor(normalize(feats, colmin, colmax)).to(device)

        ares, rmrses, lares, lrmrses = train(lr=0.003, feats=feats_input, add_data=add_data,
                                                train_idx=train_idx, large_idx=large_idx, test_idx=test_idx,
                                                num_epochs=8000, test_lrg=True, milestones=None)
        if i == 0:
            np.save(pt+'result/ares_hg_'+str(cf), np.array(ares))
            np.save(pt+'result/rmrses_hg_'+str(cf), np.array(rmrses))
            np.save(pt+'result/lares_hg_'+str(cf), np.array(lares))
            np.save(pt+'result/lrmrses_hg_'+str(cf), np.array(lrmrses))

        cfresults.append(np.array([ares[-1], rmrses[-1], lares[-1], lrmrses[-1]]))

        #np.save('fold_'+str(cf)+'_'+str(i)+'.npy', preds)

    results.append(cfresults)
    print('Fold ', cf)
    print(np.mean(cfresults, axis=0))
    print('-----------')

results = np.array(results)

print(np.mean(results.reshape(-1, 4), 0))
print(np.var(results.reshape(-1, 4), 0))

if only_hsgnn:
    np.save(pt+'result/hsgnnssl.npy', results)
elif refined:
    np.save(pt+'result/hsgnnsslglb_'+pt+'.npy', results)
elif linear:
    np.save(pt+'result/linear.npy', results)
else:
    np.save(pt+'result/mlp.npy', results)


# below 5 fold
'''

num_valid = int(0.15*(len(index)-num_test))

for cf in range(5):
    test_idx = index[cf*num_test : (cf+1)*num_test]
    train_idx = copy.deepcopy(np.append(index[(cf+1)*num_test:], index[:cf*num_test]))

    cfresults = []

    for i in range(5):
        print('Exp ', i)

        np.random.shuffle(train_idx)

        val_idx = train_idx[:num_valid]
        train_idx_ = train_idx[num_valid:]

        if only_hsgnn:
            feats_ = feats
        else:
            train_feats, colmin, colmax = normalize(feats[train_idx_], returnminmax=True)

            feats_ = torch.FloatTensor(normalize(feats, colmin, colmax)).to(device)

        cfresults.append(np.array(train(lr=0.01, feats=feats_, add_data=add_data,
                                        train_idx=train_idx_, valid_idx=val_idx, test_idx=test_idx,
                                        num_epochs=20000, echo=500, save=False, test_lrg=True, milestones=None)))

        print('--------------------------------')

    cfresults = np.stack(cfresults)
    results.append(cfresults)
    print('Fold ', cf)
    print(np.mean(cfresults, axis=0))
    print('********************************')

results = np.stack(results) # 10 fold x 10 exps x 4 metrics

print(np.mean(np.mean(results, axis=0), axis=0))

if only_hsgnn:
    np.save('91f5preH3_wo_'+pt+'.npy', results)
elif refined:
    np.save('91f5pretrain_H3_'+pt+'.npy', results)
elif linear:
    np.save('91f5LR'+pt+'.npy', results)
else:
    np.save('91f5mlp_'+pt+'.npy', results)

'''

