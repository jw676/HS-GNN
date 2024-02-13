import dgl
import numpy as np
from baselines.pointnet import *

import os


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

def eval(model, data, labels, alpha=0.01):
    model.eval()

    preds, m3s, m64s = [], [], []
    for pcd in data:
        pred, m3, m64 = model([pcd])
        preds.append(pred.detach())
        m3s.append(m3.detach())  # m3.shape == [1, 3, 3]
        m64s.append(m64.detach())

    pred = torch.stack(preds)
    m3 = torch.cat(m3s, dim=0)
    m64 = torch.cat(m64s, dim=0)
    loss, are, reg = pointnetloss(pred, labels, m3, m64, alpha=alpha)
    #are = loss_are(pred, labels).item()
    mrse = loss_mrse(pred, labels).item()
    rmrse = loss_rmrse(pred, labels).item()

    return  are.item(), mrse, rmrse

def normalize(pointcloud):
    assert len(pointcloud.shape) == 2

    norm_pointcloud = pointcloud - torch.mean(pointcloud, dim=0)
    norm_pointcloud /= torch.max(torch.norm(norm_pointcloud, dim=1))

    return norm_pointcloud

def glbnorm(pcds, maxi=None, returnmax=False):
    assert isinstance(pcds, list)
    assert len(pcds[0].shape) == 2

    pcds = [pcd - torch.mean(pcd, dim=0) for pcd in pcds]

    if maxi is not None:
        return [pcd/maxi for pcd in pcds]

    maxi = max([torch.max(torch.norm(pcd, dim=1)).item() for pcd in pcds])
    pcds = [pcd / maxi for pcd in pcds]

    if returnmax:
        return pcds, maxi

    return pcds

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.1):
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1).to(outputs.device)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1).to(outputs.device)
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    are_loss = loss_are(outputs, labels)
    reg_loss = (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)
    return are_loss + alpha * reg_loss, are_loss, reg_loss


def train(lr, pcds, num_epochs, train_idx, valid_idx, test_idx, lrg_idx, labels, echo=50, milestones=None, test_lrg=False, alpha=0.01):

    # normalization globally worse than individually

    train_pcds = [pcds[i] for i in train_idx]
    valid_pcds = [pcds[i] for i in valid_idx]
    test_pcds = [pcds[i] for i in test_idx]
    large_pcds = [pcds[i] for i in lrg_idx]

    train_labels = labels[train_idx]

    batch_size = 128

    epoch_train_idx = np.arange(len(train_pcds))
    num_batches = len(epoch_train_idx) // batch_size

    model = PointNetBatchNB().to(pcds[0].device)

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
        np.random.shuffle(epoch_train_idx)
        for batch in range(num_batches):
            batch_index = epoch_train_idx[batch * batch_size : (batch + 1) * batch_size]
            logits, m3, m64 = model([train_pcds[i] for i in batch_index])

            loss, reg, are = pointnetloss(
                outputs=logits,
                labels=train_labels[batch_index],
                m3x3=m3,
                m64x64=m64,
                alpha=alpha
            )

            loss.backward()
            optimizer.step()

        #if e % 10 == 0:
        if True:
            model.eval()

            are, mrse, rmrse = eval(model, valid_pcds, labels[valid_idx], alpha=alpha)
            ares.append(are)
            rmrses.append(rmrse)
            #losses.append(loss_val.item())
            #print('Epoch {:4d} | Valid ARE {:.2f} | Valid RMRSE {:.2f}'.format(e, are*100, rmrse*100))

            # todo: wrong best
            if are < best_are_val:
                best_are_val = are
                best_rmrse_val = rmrse
                best_epoch = e

            if e == 0:
                continue


            if are == best_are_val:

                # todo: save model
                # todo: add loss
                # todo: make a prediction result
                aret, mrset, rmrset = eval(model, test_pcds, labels[test_idx], alpha)

                best_are = aret
                best_rmrse = rmrset

                if test_lrg:
                    arel, mrsel, rmrsel = eval(model, large_pcds, labels[lrg_idx], alpha)
                    best_are_lrg = arel
                    best_rmrse_lrg = rmrsel

    print('Best Epoch {:3d}'.format(best_epoch))
    print('Best Valid ARE {:.2f} | RMRSE {:.2f}'.format(best_are_val * 100, best_rmrse_val * 100))
    print('Best Test ARE {:.2f} | RMRSE {:.2f}'.format(best_are * 100, best_rmrse * 100))
    print('Best Large Test ARE {:.2f} | RMRSE {:.2f}'.format(best_are_lrg * 100, best_rmrse_lrg * 100))

    return best_are, best_rmrse, best_are_lrg, best_rmrse_lrg

    del model
    torch.cuda.empty_cache()

if __name__=="__main__":
    loss_fn = loss_are
    device = torch.device('cuda:1')

    print(device)

    print(os.getpid())

    index = np.load('newidx.npy')

    num_test = int(0.2 * len(index))

    large_idx = np.load('lrg_test_idx.npy')


    pt = 'm'
    mapping = {'s':0, 'm':1}

    graphs, labels = dgl.load_graphs('/data/zilu/files/cntGraph')

    targets = labels['labels'][:, mapping[pt]].to(device)

    pcds = [graph.nodes['A'].data['feats'][:, :3].to(device).t() for graph in graphs] # [B, 3, n]

    pcds = [normalize(pcd).to(device) for pcd in pcds]

    valid = 5

    num_valid = int(0.15*(len(index)-num_test))

    results = []

    for cf in range(5):
        test_idx = index[cf*num_test : (cf+1)*num_test]
        train_idx = np.append(index[(cf+1)*num_test:], index[:cf*num_test])


        cfresults = []
        for i in range(5):
            print('Exp ', i)
            np.random.shuffle(train_idx)
            val_idx = train_idx[:num_valid]
            train_idx_ = train_idx[num_valid:]
            cfresults.append(np.array(train(lr=0.001, pcds=pcds, lrg_idx=large_idx, labels=targets,
                                            train_idx=train_idx_, valid_idx=val_idx, test_idx=test_idx,
                                            num_epochs=20, test_lrg=True, milestones=None, alpha=1)))
            print('--------------------------------')

        cfresults = np.stack(cfresults)
        results.append(cfresults)
        print('Fold ', cf)
        print(np.mean(cfresults, axis=0))
        print('********************************')

    results = np.stack(results) # 10 fold x 5 exps x 4 metrics

    print(np.mean(np.mean(results, axis=0), axis=0))

    np.save('91f5pointnet'+pt+'.npy', results)



