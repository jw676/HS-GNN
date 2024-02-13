import numpy as np
from dgl.data.utils import Subset
import torch
import torch.nn.functional as F
import dgl

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

def eval(pred, labels):

    are = loss_are(pred, labels).item()
    mrse = loss_mrse(pred, labels).item()
    rmrse = loss_rmrse(pred, labels).item()

    return  are, mrse, rmrse

def split_data(dataset, num_train, num_test, shuffle=True, random_state=2):
    from itertools import accumulate
    num_data = len(dataset)
    assert num_train + num_test <= num_data
    lengths = [num_train, num_test]
    if shuffle:
        indices = np.random.RandomState(seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)

    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(accumulate(lengths), lengths)]



def collate_fn_s(batch):
    graphs, features, targets = map(list, zip(*batch))
    g = dgl.batch(graphs)
    features = torch.stack(features, dim=0)
    targets = torch.stack(targets, dim=0)[:, 0]

    return g, features, targets


def collate_fn_m(batch):
    graphs, features, targets = map(list, zip(*batch))
    g = dgl.batch(graphs)
    features = torch.stack(features, dim=0)
    targets = torch.stack(targets, dim=0)[:, 1]

    return g, features, targets


def load_data(filepath):
    graphs, features = dgl.load_graphs(filepath)


    targets = features['labels']
    feats = features['feats']
    #pca = features['pca']
    #pimgs = features['pimg']

    #feats = torch.cat([glbfeats, pca], dim=1)

    print('Dataset loaded!')

    return graphs, targets, feats

def loss_fn(predictions, targets):
    mse = F.mse_loss(predictions.squeeze(), targets)
    loss = torch.sqrt(mse / torch.sum(targets ** 2))

    return loss



def loss_fn_mean(predictions, targets):
    deviation = predictions - targets

    return torch.mean((deviation / targets)**2)

def evaluate(model, test_data,  args, echo=False, save=False, test=True, valid=False, large=False, savefile=None):
    model.eval()
    predictions = []
    labels = []
    for g, feats, targets in test_data:
        if not args.add_glb:
            preds = model(g.to(args.device))
        else:
            preds = model(g.to(args.device), feats.to(args.device))

        predictions.append(preds.detach())
        labels.append(targets)

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    are, mrse, rmrse = eval(predictions.squeeze(), labels.to(args.device))

    if save and savefile is not None:
        np.save(savefile, preds.cpu().detach().numpy())
        #np.save('/data/zilu/hsgnn/newresults/edimenet_label_'+str(exp), labels.cpu().detach().numpy())

    if test and echo:
        print('Test ARE loss {:.2f}% | Test MRSE loss {:.2f}% | Test RMRSE loss {:.2f}%'.format(are*100, mrse*100, rmrse*100))
    elif valid and echo:
        print('Valid ARE loss {:.2f}% | Valid MRSE loss {:.2f}% | Valid RMRSE loss {:.2f}%'.format(are*100, mrse*100, rmrse*100))
    elif large and echo:
        print('Large ARE loss {:.2f}% | Large MRSE loss {:.2f}% | Large RMRSE loss {:.2f}%'.format(are*100, mrse*100, rmrse*100))
    elif echo:
        print('Train ARE loss {:.2f}% | Train MRSE loss {:.2f}% | Train RMRSE loss {:.2f}%'.format(are*100, mrse*100, rmrse*100))

    del preds
    torch.cuda.empty_cache()

    return are, mrse, rmrse

def evaluate_bt(model, graphs, glbfeats, labels, args):
    model.eval()
    predictions = []

    num_graphs = labels.size(0)
    num_batches = num_graphs // args.bs + 1

    for b in range(num_batches):
        start, end = b*args.bs, min((b+1)*args.bs, num_graphs)
        bgraphs = [graphs[i] for i in range(start, end)]
        bgraph = dgl.batch(bgraphs)
        feats = glbfeats[start:end, :]

        predictions.append(model(bgraph, feats))


    preds = torch.cat(predictions, dim=0)

    are, mrse, rmrse = eval(preds.squeeze(), labels)

    print('Test ARE loss {:.2f}% | Test MRSE loss {:.2f}% | Test RMRSE loss {:.2f}%'.format(are*100, mrse*100, rmrse*100))

    del preds
    torch.cuda.empty_cache()


def evaluate_bt_(model, test_data):
    model.eval()
    predictions = []
    labels = []

    for graph, feats, targets in test_data:
        predictions.append(model(graph, feats))
        labels.append(targets.unsqueeze(-1))

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    are, mrse, rmrse = eval(predictions.squeeze(), labels.squeeze())

    print('Test ARE loss {:.2f}% | Test MRSE loss {:.2f}% | Test RMRSE loss {:.2f}%'.format(are*100, mrse*100, rmrse*100))

    del predictions
    torch.cuda.empty_cache()



def evaluate_train(model, dataloader, args, exp):
    model.eval()
    preds = []
    labels = []

    bs, device = args.bs, args.device


    for graph, feats, targets in dataloader:
        if not args.add_glb:
            prediction = model(graph.to(device))
        else:
            prediction = model(graph.to(device), feats.to(device))
        preds.append(prediction.squeeze().cpu().detach().numpy())
        labels.append(targets.cpu().detach().numpy())

    preds, labels = np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)

    np.save('/data/zilu/hsgnn/results/shsgnn_train_pred_' + exp, preds)
    np.save('/data/zilu/hsgnn/results/shsgnn_train_label_' + exp, labels)

    del dataloader


def get_hidden(model, test_data,  args, exp=0, save=False, test=True, savefile=None):
    model.eval()
    hidden = []
    for g, feats, targets in test_data:
        if not args.add_glb:
            hid = model.get_hidden(g.to(args.device))
        else:
            hid = model.get_hidden(g.to(args.device), feats.to(args.device))

        hidden.append(hid.cpu().detach().numpy())
    hidden = np.stack(hidden)

    np.save(savefile, hidden)


'''

def evaluate(model, dataloader, args, exp):
    model.eval()
    preds = []
    labels = []
    device = args.device

    for graph, feats, targets in dataloader:
        if args.glb_dim == 0:
            prediction = model(graph.to(device))
        else:
            prediction = model(graph.to(device), feats.to(device))

        preds.append(prediction.squeeze())
        labels.append(targets.to(device))

    preds, labels = torch.cat(preds), torch.cat(labels)

    np.save('/data/zilu/hsgnn/results/shsgnn_test_pred_' + exp, preds.cpu().detach().numpy())
    np.save('/data/zilu/hsgnn/results/shsgnn_test_label_' + exp, labels.cpu().detach().numpy())

    loss = loss_fn_mean(preds, labels).item()

    print('Test Loss: {:.2f}%'.format(loss * 100))

    return loss

'''
def evaluate_lr(model, dataloader, device, exp):
    model.eval()
    preds = []
    labels = []

    for _, feats, targets in dataloader:
        prediction = model(feats.to(device))
        preds.append(prediction.squeeze())
        labels.append(targets)

    preds, labels = torch.cat(preds), torch.cat(labels)

    np.save('/data/zilu/hsgnn/results/0lr_test_pred_' + exp, preds.cpu().detach().numpy())
    np.save('/data/zilu/hsgnn/results/0lr_test_label_' + exp, labels.cpu().detach().numpy())

    loss = loss_fn_mean(preds, labels).item()

    print('Test Loss: {:.2f}%'.format(loss * 100))

    return loss


def test_lr(model, test_index, data, args):
    graphs, targets, glbfeats = data

    # graphs = [graph.to(args.device) for graph in graphs] # not enough gpu memory
    targets = targets.to(args.device)
    glbfeats = glbfeats.to(args.device)

    if args.predict == 's':
        targets = targets[:, 0]
    elif args.predict == 'm':
        targets = targets[:, 1]
    else:
        raise KeyError('not supported prediction target')

    model.eval()

    test_loss = model.loss_(
        glbfeats[test_index].to(args.device),
        targets[test_index].to(args.device)
    )

    print('Test loss')
    print(float(test_loss))
