from dgl.data import gdelt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

from IPython import embed

import dgl
from dgl._deprecate.graph import DGLGraph
from ogb.nodeproppred import Evaluator
from dgl_models import Net, GraphSAGE, PPRPowerIteration, SGC, DGI, Classifier, GAT
import numba
from sklearn import preprocessing
import math
import networkx as nx
from tqdm import tqdm

import utils
import argparse, pickle
import random
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F

import quadprog
from qpsolvers import solve_qp
from sklearn.manifold import TSNE
import scipy.sparse as sp


import warnings

warnings.simplefilter("ignore")

def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]

def cmd(X, X_test, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)
    
    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = [dm]
    for i in range(K-1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1,sx2,i+2))
        #scms+=moment_diff(sx1,sx2,1)
    return sum(scms)

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return (x1-x2).norm(p=2)

def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    #ss1 = sx1.mean(0)
    #ss2 = sx2.mean(0)
    return l2diff(ss1,ss2)




def cross_entropy(x, labels):
    #epsilon = 1 - math.log(2)
    y = F.cross_entropy(x, labels.view(-1), reduction="none")
    #y = torch.log(epsilon + y) - math.log(epsilon)
    #embed()
    return y

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
    #dist = torch.mm(x, y_t)
    #Ensure diagonal is zero if x=y
    #if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
def naiveIW(X, Xtest, _A=None, _sigma=1e1):
    prob =  torch.exp(- _sigma * torch.norm(X - Xtest.mean(dim=0), dim=1, p=2) ** 2 )
    for i in range(_A.shape[0]):
        prob[_A[i,:]==1] = F.normalize(prob[_A[0,:]==1], dim=0, p=1) * _A[i,:].sum()
    return prob

def MMD(X,Xtest):
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(- 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()
    return MMD_dist

def KMM(X,Xtest,_A=None, _sigma=1e1):
    #embed()
    if False:
        H = X.matmul(X.T)
        f = X.matmul(Xtest.T)
        z = Xtest.matmul(Xtest.T)
    #
    #H = torch.exp(- _sigma * pairwise_distances(X))
    #f = torch.exp(- _sigma * pairwise_distances(X, Xtest))
    #z = torch.exp(- _sigma * pairwise_distances(Xtest, Xtest))
    else:
        H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(- 1e-3 * pairwise_distances(X))
        f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(X, Xtest))
        z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
        H /= 3
        f /= 3
    #
    #embed()
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()
    
    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0],1)))
    #eps = (math.sqrt(nsamples)-1)/math.sqrt(nsamples)
    eps = 10
    #A = np.ones((2,nsamples))
    #A[1,:] = -1
    #b = np.array([[nsamples * (eps+1)], [nsamples * (eps-1)]])
    #lb = np.zeros((nsamples,1))
    #ub = np.ones((nsamples,1))*1000
    #Aeq, beq = [], []
    #embed()
    #qp_C = -A.T
    #qp_b = -b
    #meq = 0
    G = - np.eye(nsamples)
    #h = np.zeros((nsamples,1))
    #if _A is None:
    #    return None, MMD_dist
    #A = 
    #b = np.ones([_A.shape[0],1]) * 20
    _A = _A[~np.all(_A==0, axis=1)]
    b = _A.sum(1)
    h = - 0.2 * np.ones((nsamples,1))
    from cvxopt import matrix, solvers
    #return quadprog.solve_qp(H.numpy(), f.numpy(), qp_C, qp_b, meq)
    try:
        sol=solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h), matrix(_A), matrix(b))
        #sol=solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h))
    except:
        embed()
    #embed()
    #np.matmul(np.matmul(np.array(sol['x']).T, H.numpy()), sol['x']) + np.matmul(f.numpy().T, np.array(sol['x']))
    return np.array(sol['x']), MMD_dist.item()
    #return solve_qp(H.numpy(), f.numpy(), A, b, None, None, lb, ub)
    


# for connected edges
def calc_feat_smooth(adj, features):
    A = sp.diags(adj.sum(1).flatten().tolist()[0])
    D = (A - adj)
    #(D * features) ** 2
    return (D * features)
    smooth_value = ((D * features) ** 2).sum() / (adj.sum() / 2 * features.shape[1])
    
    adj_rev = 1 - adj.todense()
    np.fill_diagonal(adj_rev, 0)

    A = sp.diags(adj_rev.sum(1).flatten().tolist()[0])
    D_rev = (A - adj_rev)
    smooth_rev_value = np.power(np.matmul(D_rev, features), 2).sum() / (adj_rev.sum() / 2 * features.shape[1])
    # D = torch.Tensor(D)
    
    return smooth_value, smooth_rev_value
    #return 

@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())
'''
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())
'''
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
    js = []
    vals = []
    for i, node in enumerate(nodes):
        j, val = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        js.append(j)
        vals.append(val)
    return js, vals

def calc_emb_smooth(adj, features):
    A = sp.diags(adj.sum(1).flatten().tolist()[0])
    D = (A - adj)
    return ((D * features) ** 2).sum() / (adj.sum() / 2 * features.shape[1])

def snowball(g, max_train, ori_idxdx_train, labels):
    train_seeds = set()

    label_cnt = defaultdict(int)
    train_idxds = list(ori_idxdx_train)
    #random.shuffle(train_idxds)
    # modify the snowball sampling into a function
    train_sampler = dgl.contrib.sampling.NeighborSampler(g, 1, -1,  # 0,
                                                                neighbor_type='in', num_workers=1,
                                                                add_self_loop=False,
                                                                num_hops=2, seed_nodes=torch.LongTensor(train_idxds), 
                                                               shuffle=True)
    cnt = 0
    for __, sample in enumerate(train_sampler):
        #option 1, 
        _center_label = labels[sample.layer_parent_nid(-1).tolist()[0]]
        if _center_label < 0:
            print('here')
            continue

        _center_idxd = sample.layer_parent_nid(-1).tolist()[0]
        #mbed()
        cnt += 1
        for i in range(sample.num_layers)[::-1][1:]:
            
            for idx in sample.layer_parent_nid(i).tolist():
                if idx == _center_idxd or labels[idx].item() < 0 or labels[idx].item() != _center_label.item():
                    continue
                if idx not in train_seeds and label_cnt[labels[idx].item()] < max_train[labels[idx].item()] and idx in ori_idxdx_train:
                    train_seeds.add(idx)
                    label_cnt[labels[idx].item()] += 1
        if __ > 1000:
            break
            print('node{}, layer {}'.format(__, i), sum(label_cnt.values()))
        #print(label_cnt)
        #if cnt == 5:
        #    break
        #print("iter", sample.layer_parent_nid(5))
        #init_labels = Counter(labels[list(train_seeds)])
        #if len(label_cnt.keys()) == num_class and min(label_cnt.values()) == max_train:
        done = True
        for k in range(labels.max().item()+1):
            try:
                if label_cnt[k] < max_train[k]:
                    done = False
                    break
            except:
                embed()
        if done:
            break
    print("number of seed used:{}".format(cnt))
    return train_seeds, cnt

def output_edgelist(g, OUT):
    for i,j in zip(g.edges()[0].tolist(), g.edges()[1].tolist()):
        OUT.write("{} {}\n".format(i, j))

def read_posit_emb(IN):
    tmp = IN.readline()
    a, b = tmp.strip().split(' ')
    emb = torch.zeros(int(a),int(b))
    for line in IN:
        tmp = line.strip().split(' ')
        emb[int(tmp[0]), :] = torch.FloatTensor(list(map(float, tmp[1:])))
    return emb

def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_idxnvsqrt_corr = 1 / np.sqrt(D_vec)
    D_idxnvsqrt_corr = sp.diags(D_vec_idxnvsqrt_corr)
    return D_idxnvsqrt_corr @ A @ D_idxnvsqrt_corr
    
def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_idxnner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_idxnner.toarray())

@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals

def creatCSR(ppr_sparse):
    indptr = [0]
    indices = []
    data = []
    for ind, val in zip(*ppr_sparse):
        indices += ind
        data += val
        assert len(ind) == len(val)
        indptr.append(indptr[-1]+len(ind))
    return data, indices, indptr

def calc_ppr_appr(adj_matrix: sp.spmatrix, alpha: float, idx) -> np.ndarray:
    
    
    for _idxd in idx.tolist():
        s = np.ones((1, adj_matrix.shape[1])) / adj_matrix.shape[1]
        ppr = s
        for _ in range(10):
            
            s = (1-alpha) * s @ adj_matrix
            ppr += s
        #embed()  
        
    return preds
def dgi(args, new_classes):
    # training params
    batch_size = 1
    nb_epochs = 10000
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    drop_prob = 0.0
    hid_units = 128
    sparse = True
    # unk = True, if we have unseen classes
    unk = False
    nonlinearity = 'prelu' # special name to separate parameters

    torch.cuda.set_device(args.gpu)

    if args.dataset == 'ogbn-arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name = args.dataset)
        g, labels = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        features = g.ndata['feat']
        adj = g.adjacency_matrix_scipy()
        min_max_scaler = preprocessing.MinMaxScaler()
        feat = min_max_scaler.fit_transform(features)
        # feat_smooth_matrix = calc_feat_smooth(adj, feat)

        g = g.remove_self_loop().add_self_loop()
        old_g = DGLGraph(g.to_networkx())
        old_g.readonly()
        
        evaluator = Evaluator(name="ogbn-arxiv")
        split_idxdx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idxdx["train"], split_idxdx["valid"], split_idxdx["test"]
        
        #adj_train = g.subgraph(idx_train).adjacency_matrix_scipy()
    elif args.dataset == 'reddit':
        from dgl.data import RedditDataset
        import os
        os.environ["DGL_DOWNLOAD_DIR"] = "/DATA/"
        dataset = RedditDataset(self_loop=True)
        g = dataset[0]
        adj = g.adjacency_matrix_scipy()
        labels = g.ndata['label']
        num_classes = dataset.num_classes
        # get node feature
        features = g.ndata['feat']
        # get data split
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        idx_train = train_mask.nonzero().view(-1)
        idx_val = val_mask.nonzero().view(-1)
        idx_test = test_mask.nonzero().view(-1)
    
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    
    labels = torch.LongTensor(labels)
    nb_classes = max(labels).item() + 1

    #
    xent = nn.CrossEntropyLoss(reduction='none')
    #xent = nn.CrossEntropyLoss()

    cnt_wait = 0
    best = 1e9
    best_t = 0

    if args.gpu >= 0:
        print('Using CUDA')
        g = g.to(torch.device('cuda:{}'.format(args.gpu)))
        features = features.cuda()
        labels = labels.cuda()

    

    
    best_val_acc = 0
    cnt_wait = 0
    finetune = False
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []
    
    #print("original length:{}".format(len(ori_idxdx_train)))
    
    num_seeds = []
    all_runs_data = defaultdict(list)
    feature_smoothness = []
    embedding_smoothness = []
    avg_dist, max_dist = [], []
    #pre-compute stage
    #if args.snowball_sample:
    if True:
        out_degree = np.sum(adj, axis=1).A1
        nnodes = adj.shape[0]

    avg_mmd_dist = []

    pickle_seeds = []
    label_ratio = args.label_ratio
    idx_train_list = idx_train.view(-1).cpu().numpy().tolist()
    
    for _run in range(args.n_repeats):
        
        if args.biased_sample:
           
            if True:
                train_seeds = pickle.load(open('data/{}_{}.p'.format(args.dataset, label_ratio), 'rb'))[_run]
                #embed()
                #train_seeds = pickle.load(open('ogbn-ppr-idx-526.p', 'rb'))[_run]
                
                idx_train = torch.LongTensor(list(train_seeds))
                max_train = dict(Counter(labels[idx_train].view(-1).tolist()))
                print('number of classes {}'.format(len(max_train)))
                if True:
                    new_idxdx_val, new_idxdx_test = [], []
                    for idx in idx_val.tolist():
                        if labels[idx].item() in max_train:
                            new_idxdx_val.append(idx)
                    for idx in idx_test.tolist():
                        if labels[idx].item() in max_train:
                            new_idxdx_test.append(idx)
                    idx_val, idx_test = torch.LongTensor(new_idxdx_val), torch.LongTensor(new_idxdx_test)
            elif True:
                result = calc_ppr_topk_parallel(adj.indptr, adj.indices, out_degree, numba.float32(0.1), numba.float32(0.00005), idx_train_list, topk=200)
                idx_train = train_mask.nonzero().view(-1)
                idx_train_list = labels[idx_train].view(-1).cpu().numpy().tolist()
                max_train = dict(Counter(labels[idx_train].view(-1).tolist())) 
                idx_train_list = set(idx_train.view(-1).cpu().numpy().tolist())
                label_cnt = dict()
                total_cnt = 0
                import math
                for k in max_train:
                    max_train[k] = math.floor( label_ratio* max_train[k])
                    #if max_train[k] > label_ratio * label_ratio*g.number_of_nodes():
                    #print(max_train[k])
                    if max_train[k] >= 10:
                        
                        label_cnt[k] = 0
                        total_cnt+= max_train[k]
                    else:
                        max_train[k] = 0
                print("sum of max train {}/{}, num of class {}".format(sum(max_train.values()), total_cnt, len(label_cnt)))
                train_seeds = set()
                #
                ppr_list = list(zip(result[0], result[1]))
                np.random.shuffle(ppr_list)
                #
                seed_cnt = 0 
                for a,b in ppr_list:
                    #
                    seed_cnt += 1
                    idx = a[-1]
                    #embed()
                    if len(a.tolist()) <= 50 or labels[idx].item() not in label_cnt:
                    #if labels[idx].item() not in label_cnt:
                        continue
                    #max_n = max(max_n, len(result[0][i].tolist()))
                    #continue
                    #train_seeds.add(idx)
                    for _idx,val in zip(a.tolist(), b.tolist()):
                        #if val < 0.05:
                        #    continue
                        
                        if labels[_idx].item() == labels[idx].item() and label_cnt[labels[_idx].item()] < max_train[labels[_idx].item()] and _idx not in train_seeds:
                        #if _idx in idx_train_list and labels[_idx].item() in label_cnt and label_cnt[labels[_idx].item()] < max_train[labels[_idx].item()] and _idx not in train_seeds:
                            #if len(train_seeds) % 100 == 0:
                            #    print("{}/{}".format(len(train_seeds), label_ratio * g.number_of_nodes()))
                            train_seeds.add(_idx)
                            label_cnt[labels[_idx].item()] += 1
                    #embed()
                    #print(sum(label_cnt.values()))
                    done = True
                    for k in label_cnt:
                        if label_cnt[k] < max_train[k]:
                            done = False
                            break
                    if done:
                        print('break!')
                        break

                print('numeber of seeds used {}'.format(seed_cnt), len(train_seeds), total_cnt)
                if not done:
                    for idx in idx_train_list:
                        if labels[idx].item() in label_cnt and label_cnt[labels[idx].item()] < max_train[labels[idx].item()] and idx not in train_seeds:
                            train_seeds.add(idx)
                            label_cnt[labels[idx].item()] += 1
                        done = True
                        for k in label_cnt:
                            if label_cnt[k] < max_train[k]:
                                done = False
                                break
                        if done:
                            break

                # random sample to satisfy the training distribution


                #embed()
                #print('numeber of seeds used {}'.format(seed_cnt), len(train_seeds), total_cnt)
                pickle_seeds.append(train_seeds)
                idx_train = torch.LongTensor(list(train_seeds))
                #print(list(train_seeds)[:10])
                #embed()
                continue
                #
                
                #continue
                
                new_idxdx_val, new_idxdx_test = [], []
                for idx in idx_val.tolist():
                    if labels[idx].item() in label_cnt:
                        new_idxdx_val.append(idx)
                for idx in idx_test.tolist():
                    if labels[idx].item() in label_cnt:
                        new_idxdx_test.append(idx)
                idx_val, idx_test = torch.LongTensor(new_idxdx_val), torch.LongTensor(new_idxdx_test)
            
            if args.arch in [3,5]:
                perm = torch.randperm(idx_test.shape[0])
                sub_idx = perm[:len(idx_train)]
                sub_idx_test = idx_test[sub_idx]

                if False:
                    ppr_sparse = calc_ppr(adj.indptr, adj.indices, out_degree, numba.float32(0.1), numba.float32(0.001), idx_train.tolist())
                    
                    ppr_train = sp.csr_matrix(creatCSR(ppr_sparse), (idx_train.shape[0], g.number_of_nodes()))
                    ppr_sparse = calc_ppr(adj.indptr, adj.indices, out_degree, numba.float32(0.1), numba.float32(0.001), sub_idxdx_test.tolist())
                    ppr_test = sp.csr_matrix(creatCSR(ppr_sparse), (idx_train.shape[0], g.number_of_nodes()))
                    
                    ppr_train = torch.FloatTensor(ppr_train.todense())
                    ppr_test = torch.FloatTensor(ppr_test.todense())
                else:
                    ppr_train = torch.FloatTensor(adj[idx_train.tolist(), :].todense())
                    ppr_test = torch.FloatTensor(adj[sub_idx_test.tolist(), :].todense())
                #embed()
                label_balance_constraints = np.zeros((labels.max().item()+1, len(idx_train)))
                for i, idx in enumerate(idx_train):
                    label_balance_constraints[labels[idx], i] = 1
                #embed()
                kmm_weight, MMD_dist = KMM(ppr_train, ppr_test, label_balance_constraints)
                print(kmm_weight.max(), kmm_weight.min())
            print(idx_train.shape)

        else:
            if False:
                train_seeds = pickle.load(open('ogbn-ppr-idx-526.p', 'rb'))[_run]
                #embed()
                max_train = dict(Counter(labels[list(train_seeds)].view(-1).tolist()))
                print('number of classes {}'.format(len(max_train)))
                new_idxdx_val, new_idxdx_test = [], []
                for idx in idx_val.tolist():
                    if labels[idx].item() in max_train:
                        new_idxdx_val.append(idx)
                for idx in idx_test.tolist():
                    if labels[idx].item() in max_train:
                        new_idxdx_test.append(idx)
                idx_val, idx_test = torch.LongTensor(new_idxdx_val), torch.LongTensor(new_idxdx_test)

                perm = torch.randperm(idx_train.shape[0])
                #label_ratio = 0.01
                sub_idxdx = perm[:len(train_seeds)]
                
                idx_train = idx_train[sub_idxdx]
                #idx_train = idx_train[knn_idxdx]
                if args.arch in [3,5]:
                    perm = torch.randperm(idx_test.shape[0])
                    sub_idxdx = perm[:len(train_seeds)]
                    sub_idxdx_test = idx_test[sub_idxdx]
                    if True:
                        ppr_sparse = calc_ppr(adj.indptr, adj.indices, out_degree, numba.float32(0.1), numba.float32(0.001), idx_train.tolist())
                        
                        ppr_train = sp.csr_matrix(creatCSR(ppr_sparse), (idx_train.shape[0], g.number_of_nodes()))
                        ppr_sparse = calc_ppr(adj.indptr, adj.indices, out_degree, numba.float32(0.1), numba.float32(0.001), sub_idxdx_test.tolist())
                        ppr_test = sp.csr_matrix(creatCSR(ppr_sparse), (idx_train.shape[0], g.number_of_nodes()))
                        
                        ppr_train = torch.FloatTensor(ppr_train.todense())
                        ppr_test = torch.FloatTensor(ppr_test.todense())
                    else:
                        ppr_train = torch.FloatTensor(adj[idx_train.tolist(), :].todense())
                        ppr_test = torch.FloatTensor(adj[sub_idxdx_test.tolist(), :].todense())
                    label_balance_constraints = np.zeros((labels.max().item()+1, len(idx_train)))
                    for i, idx in enumerate(idx_train):
                        label_balance_constraints[labels[idx], i] = 1
                    #embed()
                    kmm_weight, MMD_dist = KMM(ppr_train, ppr_test, label_balance_constraints)
            else:
                idx_train = train_mask.nonzero().view(-1)
                perm = torch.randperm(idx_train.shape[0])
                #embed()
                sub_idxdx = perm[:int(label_ratio*idx_train.shape[0])]
                idx_train = idx_train[sub_idxdx]
                label_cnt = Counter(labels[idx_train].cpu().tolist())
                new_idxdx_val, new_idxdx_test = [], []
                for idx in idx_val.tolist():
                    if labels[idx].item() in label_cnt and label_cnt[labels[idx].item()] >= 10:
                        new_idxdx_val.append(idx)
                for idx in idx_test.tolist():
                    if labels[idx].item() in label_cnt and label_cnt[labels[idx].item()] >= 10:
                        new_idxdx_test.append(idx)
                idx_val, idx_test = torch.LongTensor(new_idxdx_val), torch.LongTensor(new_idxdx_test)
                print("number of class:{}".format( len(Counter(labels[idx_test].cpu().tolist()))))
                
            test_lbls = labels[idx_test]
        perm = torch.randperm(idx_test.shape[0])
        iid_train = idx_test[perm[:idx_train.shape[0]]] 
        
        train_lbls = labels[idx_train]

        if args.gnn_arch == 'graphsage':
            model = GraphSAGE(g,
                    ft_size,
                    args.n_hidden,
                    nb_classes,
                    args.n_layers,
                    #F.relu,
                    F.relu,
                    args.dropout,
                    args.aggregator_type
                    )
        elif args.gnn_arch == 'gat':
            model = GAT(g,
                    ft_size,
                    args.n_hidden,
                    nb_classes,
                    args.n_layers,
                    F.elu,
                    args.dropout
                    )
        elif args.gnn_arch == 'ppnp':
            model = PPRPowerIteration(ft_size, args.n_hidden, nb_classes, adj, alpha=0.1, niter=10, drop_prob=args.dropout)
        elif args.gnn_arch == 'sgc':
            model = SGC(g,
                    ft_size,
                    args.n_hidden,
                    nb_classes,
                    args.n_layers,
                    F.tanh,
                    args.dropout,
                    train_mask
                    )
        else:
            model = Net(g,
                    ft_size,
                    args.n_hidden,
                    nb_classes,
                    args.n_layers,
                    #F.relu,
                    F.tanh,
                    args.dropout,
                    args.aggregator_type
                    )

        #optimiser = torch.optim.Adam([{'params': model.fcs[0].parameters(), 'weight_decay':args.weight_decay}, {'params': model.fcs[1].parameters(), 'weight_decay':0}], lr=args.lr)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #print(optimiser)
        print("leng of train", len(idx_train))
        if args.gpu >= 0:
            model.cuda()
        #train_loader = DataLoader(idx_train, batch_size = 200, shuffle=True)
        best_acc, best_epoch = 0.0, 0.0
        #torch.autograd.set_detect_anomaly(True)
        plot_x, plot_y, plot_z = [], [], []
        #np.random.shuffle(kmm_weight)
        
        for epoch in range(args.n_epochs):
            if args.arch == 4 and epoch % 20 == 1:
            #    Z = ppr_vector.matmul(model.h.detach().cpu())
                kmm_weight, MMD_dist = KMM(model.h[idx_train, :].detach().cpu(), model.h[idx_test, :].detach().cpu(), label_balance_constraints)
            #else:
            #    kmm_weight = None
            #

            model.train()
            optimiser.zero_grad()

            if args.dataset != 'ogbn-arxiv':
                logits = model(features)
                loss = xent(logits[idx_train], labels[idx_train])
            else:
                logits = model(features, bns=args.bn)
                loss = cross_entropy(logits[idx_train], labels[idx_train])

            if args.arch == 0:
                loss = loss.mean()
                total_loss = loss
            elif args.arch == 1:
                loss = loss.mean()
                total_loss = loss + 0.1 * MMD(model.h[idx_train, :], model.h[iid_train, :])
            elif args.arch == 2:
                loss = loss.mean()
                total_loss = loss + 0.1 * cmd(model.h[idx_train, :], model.h[iid_train, :])
            elif args.arch in [3,4]:
                loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean()
                #total_loss = loss
                total_loss = loss +  0.1 * cmd(model.h[idx_train, :], model.h[iid_train, :])
            elif args.arch == 5:
                loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean()
                total_loss = loss

            total_loss.backward()
            optimiser.step()
            with torch.no_grad():
                if epoch % 10 == 0:
                
                    model.eval()
                    logits = model(features, bns=args.bn)
                    #logits = model(features, True)
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds[idx_train] == train_lbls.view(-1)).sum().float().item() / preds[idx_train].shape[0]
                    val_acc = (preds[idx_val] == labels[idx_val].view(-1)).sum().float().item() / preds[idx_val].shape[0]

                    if val_acc > best_acc:
                        torch.save(model.state_dict(), 'best_gnn_large.pkl')
                        best_acc = val_acc
                        best_epoch = epoch
                    test_acc = (preds[idx_test] == labels[idx_test].view(-1)).sum().float().item() / preds[idx_test].shape[0]
                    if args.gnn_arch == 'gat':
                        cmd_test = 0.1
                    else:
                        cmd_test = cmd(model.h[idx_train, :], model.h[idx_test, :]).item()
                    
                    
                    print("epoch:{}, loss:{}, cmd:{}, train acc:{}, valid acc:{}, test acc:{} ".format(epoch, loss.item(), cmd_test, acc, val_acc, test_acc))
                    
        if cmd_test > 0:
            print("cmd:{}, best epoch:{}, best validation acc:{}".format(cmd_test, best_epoch, best_acc))
            #print(_run)
            #pickle_runs.append([_run, cmd_test, best_acc])
        model.load_state_dict(torch.load('best_gnn_large.pkl'))
        model.eval()

        embeds = model(features, bns = args.bn).detach()
        
        logits = embeds[idx_test]
        preds_all = torch.argmax(embeds, dim=1)
        embeds = embeds.cpu()

        if False:
            min_max_scaler = preprocessing.MinMaxScaler()
            emb = min_max_scaler.fit_transform(embeds.cpu().numpy())
        else:
            emb = embeds.cpu().numpy()
            #emb = pos_emb.numpy()
        micro_f1.append(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro'))
        macro_f1.append(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='macro'))
    return micro_f1, macro_f1, avg_mmd_dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    # register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--gnn-arch", type=str, default='gcn',
                        help="gnn arch of gcn/gat/graphsage")
    parser.add_argument("--SR", type=bool, default=False,
                        help="use shift-robust or not")
    parser.add_argument("--arch", type=int, default=0,
                        help="use which variant of the model")
    parser.add_argument("--biased-sample", type=bool, default=False,
                        help="use biased (non IID) training data")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-out", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="Weight for L2 loss")
    parser.add_argument("--label-ratio", type=float, default=0.01,
                        help="label ratio 1%, 5%")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="print verbose step-wise information")
    parser.add_argument("--n-repeats", type=int, default=20,
                        help=".")
    parser.add_argument("--bn", type=bool, default=False,
                        help="print verbose step-wise information")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--dataset',type=str, default='cora')
    parser.add_argument('--sc', type=float, default=0.0, help='GCN self connection')
    args = parser.parse_args()
    #
    #
    #
    #torch.manual_seed(7)
    #np.random.seed(7)
    if args.dataset == 'cora':
        num_class = 7
    elif args.dataset == 'citeseer':
        num_class = 6
    elif args.dataset == 'ppi':
        num_class = 9
    elif args.dataset == 'dblp':
        num_class = 5
    # 3 both techniques, 2 regularization only, 0 vanilla model
    if args.SR:
        args.arch = 3
    else:
        args.arch = 0
        #print(arch)
        #if arch != 0 and arch != 2:
        #    continue
    #args.arch = 3
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []
    #for i in utils.generateUnseen(num_class, args.num_unseen):
    micro_f1, macro_f1, out_acc = dgi(args, [])
    torch.cuda.empty_cache()
    # embed()
    #print(np.mean(in_acc), np.std(in_acc), np.mean(out_acc), np.std(out_acc))
    print("sr arch {}:".format(args.arch), np.mean(micro_f1), np.std(micro_f1), np.mean(macro_f1), np.std(macro_f1))
    #print(out_acc)
    #plt.scatter(out_acc, micro_f1)
    #plt.scatter(X_embedded[idx_train, 0], X_embedded[idx_train, 1], 10 * kmm_weight)
    #plt.savefig('{}_{}_cmd.png'.format(args.dataset, args.gnn_arch))
    #break
