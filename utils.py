from collections import defaultdict, Counter
import random
import torch
import numpy as np
from IPython import embed
import scipy.sparse as sp
import networkx as nx
import sys
import pickle as pkl
from itertools import combinations 

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def generateUnseen(num_class, num_unseen):
    return combinations(range(num_class), num_unseen)

def load_data_dblp(args):
    dataset = args.dataset
    metapaths = args.metapaths
    sc = args.sc

    if dataset == 'acm':
        data = sio.loadmat('data/{}.mat'.format(dataset))
    else:
        data = pkl.load(open('data/{}.pkl'.format('dblp_v8_reducedlabel_20'), "rb"))
    label = data['label']
    N = label.shape[0]

    truefeatures = data['feature'].astype(float)
    
    rownetworks = [data[metapath] + np.eye(N)*sc for metapath in metapaths]
    # embed()
    rownetworks = [sp.csr_matrix(rownetwork) for rownetwork in rownetworks]

    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    truefeatures_list = []
    for _ in range(len(rownetworks)):
        truefeatures_list.append(truefeatures)

    return rownetworks, truefeatures_list, label, idx_train, idx_val, idx_test

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
            #if True:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    # embed()
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    # embed()
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #embed()
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    #embed()
    #idx_train = range(len(y))
    if dataset_str == 'pubmed':
        idx_train = range(10000)
    elif dataset_str == 'cora':
        idx_train = range(1500)
    else:
        idx_train = range(1000)
    idx_val = range(len(y), len(y)+500)
    return adj, features, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def createTraining(labels, max_train=200, balance=True, new_classes=[]):
    dist = defaultdict(list)
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)

    for idx,l in enumerate(labels.numpy().tolist()[:max_train]):
        dist[l].append(idx)
    # print(dist)
    cat = []
    _sum = 0
    if balance:
        for k in dist:
            if k in new_classes:
                continue
            _sum += len(dist[k])
            # cat += random.sample(dist[k], k=15)
            train_mask[random.sample(dist[k], k=3)] = 1
    for k in new_classes:
       train_mask[random.sample(dist[k], k=3)] = 1 
    # print(_sum, sum(train_mask))
    return train_mask
    # print(len(set(cat)))

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def ind_normalize_adj(adj):
   # """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()
    #return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

import itertools
def createDBLPTraining(labels, idx_train, idx_val, idx_test, max_train=20, balance=True, new_classes=[], unknown=False):
    
    labels = [np.where(r==1)[0][0] if r.sum() > 0 else -1 for r in labels]
    #print(Counter(labels))
    new_mapping = {}
    dist = defaultdict(list)
    new_idx_train, new_idx_val, in_idx_test, out_idx_test, new_idx_test = [], [], [], [], []
    
    for idx in idx_train:
        dist[labels[idx]].append(idx)

    for k in range(len(dist)):
        if k not in new_classes:
            new_mapping[k] = len(new_mapping)
    # embed()
    if False:
        for idx in idx_train:
            if labels[idx]:
                #unknown label id
                new_idx_train.append(idx)
    else:
        for k in dist:
            # embed()
            if max_train < len(dist[k]):
                new_idx_train += np.random.choice(dist[k], max_train, replace=False).tolist()
            else:
                new_idx_train += dist[k]
            # print(len(set(new_idx_train)))


    for idx in idx_val:
        if labels[idx] in new_mapping:
            #unknown label id
            new_idx_val.append(idx)
        else:
            new_idx_val.append(idx)
    
    for idx in idx_test:
        if labels[idx] in new_mapping:
            #unknown label id
            new_idx_test.append(idx)
            in_idx_test.append(idx)
        else:
            #unknown class
            if unknown:
                new_idx_test.append(idx)
                out_idx_test.append(idx)
    
    for idx,label in enumerate(labels):
        if label < 0:
            continue
        if label in new_mapping:
            labels[idx] = new_mapping[label]
        else:
            labels[idx] = len(new_mapping)
    #print('its here')
    # embed()
    return new_idx_train, new_idx_val, in_idx_test, new_idx_test, out_idx_test, labels

def createPPITraining(train_labels, val_labels, test_labels, idx_train, idx_val, idx_test, new_classes=[], unknown=False):
    # labels = [np.where(r==1)[0][0] for r in labels]
    new_mapping = {}
    dist = defaultdict(list)
    new_idx_train, new_idx_val, in_idx_test, out_idx_test, new_idx_test = [], [], [], [], []
    
    for idx in idx_train:
        dist[train_labels[idx]].append(idx)
    
    for k in range(len(dist)):
        if k not in new_classes:
            new_mapping[k] = len(new_mapping)

    for idx in idx_train:
        assert train_labels[idx] > -1
        if train_labels[idx] in new_mapping:
            #unknown label id
            new_idx_train.append(idx)
            train_labels[idx] = new_mapping[train_labels[idx]]
        else:
            train_labels[idx] = len(new_mapping)

    for idx in idx_val:
        assert val_labels[idx] > -1
        if val_labels[idx] in new_mapping:
            #unknown label id
            new_idx_val.append(idx)
            val_labels[idx] = new_mapping[val_labels[idx]]
        else:
            new_idx_val.append(idx)
            val_labels[idx] = len(new_mapping)
    
    for idx in idx_test:
        assert test_labels[idx] > -1
        if test_labels[idx] in new_mapping:
            #unknown label id
            new_idx_test.append(idx)
            in_idx_test.append(idx)
            test_labels[idx] = new_mapping[test_labels[idx]]
        else:
            #unknown class
            test_labels[idx] = len(new_mapping)
            if unknown:
                new_idx_test.append(idx)
                out_idx_test.append(idx)
    

    print(new_mapping)
    return new_idx_train, new_idx_val, in_idx_test, out_idx_test, new_idx_test

def createClusteringData(labels, idx_train, idx_val, idx_test, max_train=200, balance=True, new_classes=[], unknown=False):
    labels = [np.where(r==1)[0][0] for r in labels]
    #
    # np.where(labels==1)
    new_mapping = {}
    
    dist = defaultdict(list)

    for idx in idx_train:
        dist[labels[idx]].append(idx)
        # train_mask[idx] = 1
    
    for k in range(len(dist)):
        if k not in new_classes:
            new_mapping[k] = len(new_mapping)
    print('new mapping is {}'.format(new_mapping))
    new_idx_train, new_labels = [], []
    
    # embed()
    for idx in idx_train:
        if labels[idx] not in new_classes:
            new_idx_train.append(idx)
    
    for idx,label in enumerate(labels):
        if label in new_mapping:
            new_labels.append(new_mapping[label])
        else:
            new_labels.append(len(new_mapping))


    # print(dist)
    # embed()
    return new_idx_train, labels, new_labels
