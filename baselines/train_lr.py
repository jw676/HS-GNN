import copy

import numpy as np
import torch
from baselines.nnutil import MLPG

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

def train(lr, feats, add_data, num_epochs, train_idx, valid_idx, test_idx, echo=50, save=False, milestones=None, test_lrg=False):

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


    ares, rmrses, losses = [], [], []

    best_are, best_rmrse = 1e10, 1e10
    best_are_lrg, best_rmrse_lrg = 1e10, 1e10
    best_are_val, best_rmrse_val = 1e10, 1e10
    best_epoch = None

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


        if e % valid == 0:
            model.eval()

            are, mrse, rmrse = eval(model, feats[valid_idx], strength[valid_idx], add_data[valid_idx], refined)
            ares.append(are)
            rmrses.append(rmrse)
            #losses.append(loss_val.item())

            # todo: wrong best
            if are < best_are_val:
                best_are_val = are
                best_rmrse_val = rmrse
                best_epoch = e
                bestmodel = copy.deepcopy(model)

            if e == 0:
                continue

            #if e > 1000 and (best_are_val < np.array(ares[-10:])).all() and (best_rmrse_val < np.array(rmrses[-10:])).all():
            #if e > 1000 and not (best_are_val==np.array(ares[-5:])).any() and early_stop(best_are_val, ares[-10:-1]) and early_stop(best_rmrse_val, rmrses[-10:-1]):
            if e > 1000 and not (best_are_val == np.array(ares[-10:])).any() and early_stop(
                    best_are_val, ares[-20:]) and early_stop(best_rmrse_val, rmrses[-20:]):

                # todo: early stop criteria
                print('Early stop at epoch ' + str(e))
                break


            if e == best_epoch:

                # todo: save model
                # todo: add loss
                # todo: make a prediction result
                are, mrse, rmrse = eval(model, feats[test_idx], strength[test_idx], add_data[test_idx], refined)

                best_are = are
                best_rmrse = rmrse

                if test_lrg:
                    are, mrse, rmrse = eval(model, feats[large_idx], strength[large_idx], add_data[large_idx], refined)
                    best_are_lrg = are
                    best_rmrse_lrg = rmrse


    bestmodel.eval()
    if refined:
        preds = model(feats, add_data).cpu().detach().numpy()
    else:
        preds = model(feats).cpu().detach().numpy()


    print('Best epoch {:5d}'.format(best_epoch))
    print('Best Valid ARE {:.2f} | RMRSE {:.2f}'.format(best_are_val * 100, best_rmrse_val * 100))
    print('Best Test ARE {:.2f} | RMRSE {:.2f}'.format(best_are * 100, best_rmrse * 100))
    print('Best Large Test ARE {:.2f} | RMRSE {:.2f}'.format(best_are_lrg * 100, best_rmrse_lrg * 100))

    del model, bestmodel
    torch.cuda.empty_cache()

    return best_are, best_rmrse, best_are_lrg, best_rmrse_lrg, preds


loss_fn = loss_rmrse
device = torch.device('cuda:5')

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

nontube = True

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
    other_idx = copy.deepcopy(np.append(index[(cf+1)*num_test:], index[:cf*num_test]))

    # num_valid == num_test
    val_idx = other_idx[:num_test]
    train_idx = other_idx[num_test:]

    cfresults = []

    for i in range(5):
        print('Exp ', i)
        np.random.shuffle(train_idx)

        if only_hsgnn:
            feats_input = feats
        else:
            _, colmin, colmax = normalize(feats[train_idx], returnminmax=True)

            feats_input = torch.FloatTensor(normalize(feats, colmin, colmax)).to(device)

        are, rmrse, lare, lrmrse, preds = train(lr=0.003, feats=feats_input, add_data=add_data,
                                                train_idx=train_idx, valid_idx=val_idx, test_idx=test_idx,
                                                num_epochs=8000, echo=500, save=False, test_lrg=True, milestones=None)

        cfresults.append(np.array([are, rmrse, lare, lrmrse]))

        #np.save('fold_'+str(cf)+'_'+str(i)+'.npy', preds)

    results.append(cfresults)
    print('Fold ', cf)
    print(np.mean(cfresults, axis=0))
    print('-----------')

results = np.array(results)

print(np.mean(results.reshape(-1, 4), 0))
print(np.var(results.reshape(-1, 4), 0))

if only_hsgnn:
    np.save('91f10preH3_wo_rmrse2_'+pt+'.npy', results)
elif refined:
    np.save('91f10pretrain_H3_41_ncnt_'+pt+'.npy', results)
elif linear:
    np.save('91f10LR_rmrse2_'+pt+'.npy', results)
else:
    np.save('91f10mlp_rmrse2_ncnt_'+pt+'.npy', results)


# below 5 fold
