import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

from IPython import embed
import dgl
from ogb.nodeproppred import Evaluator
from dgl_models import Net, GraphSAGE, PPRPowerIteration, SGC, GAT

from sklearn import preprocessing
import networkx as nx

import utils
import argparse, pickle
from sklearn.metrics import f1_score

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
    return torch.mean(y)

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
    b = np.ones([_A.shape[0],1]) * 20
    h = - 0.2 * np.ones((nsamples,1))
    
    from cvxopt import matrix, solvers
    #return quadprog.solve_qp(H.numpy(), f.numpy(), qp_C, qp_b, meq)
    try:
        solvers.options['show_progress'] = False
        sol=solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h), matrix(_A), matrix(b))
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

def calc_emb_smooth(adj, features):
    A = sp.diags(adj.sum(1).flatten().tolist()[0])
    D = (A - adj)
    return ((D * features) ** 2).sum() / (adj.sum() / 2 * features.shape[1])

def snowball(g, max_train, ori_idx_train, labels):
    train_seeds = set()

    label_cnt = defaultdict(int)
    train_ids = list(ori_idx_train)
    #random.shuffle(train_ids)
    # modify the snowball sampling into a function
    train_sampler = dgl.contrib.sampling.NeighborSampler(g, 1, -1,  # 0,
                                                                neighbor_type='in', num_workers=1,
                                                                add_self_loop=False,
                                                                num_hops=2, seed_nodes=torch.LongTensor(train_ids), 
                                                               shuffle=True)
    cnt = 0
    for __, sample in enumerate(train_sampler):
        #option 1, 
        _center_label = labels[sample.layer_parent_nid(-1).tolist()[0]]
        if _center_label < 0:
            print('here')
            continue

        _center_id = sample.layer_parent_nid(-1).tolist()[0]
        #mbed()
        cnt += 1
        for i in range(sample.num_layers)[::-1][1:]:
            for idx in sample.layer_parent_nid(i).tolist():
                if idx == _center_id or labels[idx].item() < 0 or labels[idx].item() != _center_label.item():
                    continue
                if idx not in train_seeds and label_cnt[labels[idx].item()] < max_train[labels[idx].item()] and idx in ori_idx_train:
                    train_seeds.add(idx)
                    label_cnt[labels[idx].item()] += 1
                
        #print(label_cnt)
        #if cnt == 5:
        #    break
        #print("iter", sample.layer_parent_nid(5))
        #init_labels = Counter(labels[list(train_seeds)])
        #if len(label_cnt.keys()) == num_class and min(label_cnt.values()) == max_train:
        done = True
        for k in range(labels.max().item()+1):
            if label_cnt[k] < max_train[k]:
                done = False
                break
        if done:
            break
    # print("number of seed used:{}".format(cnt))
    #print(label_cnt)
    return train_seeds, cnt
    # labels problem
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
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr
    
def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())


def main(args, new_classes):
    # unk = True, if we have unseen classes
    unk = False
    nonlinearity = 'prelu' # special name to separate parameters

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, one_hot_labels, ori_idx_train, idx_val, idx_test = utils.load_data(args.dataset)
        labels = [np.where(r==1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]
        #idx_train, idx_val, in_idx_test, idx_test, out_idx_test, labels = utils.createTraining(one_hot_labels, ori_idx_train, idx_val, idx_test, new_classes=new_classes, unknown=unk)
        features = torch.FloatTensor(utils.preprocess_features(features))
    

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        # important to add self-loop
        min_max_scaler = preprocessing.MinMaxScaler()
        #feat = min_max_scaler.fit_transform(features)
        feat = F.normalize(features, p=1,dim=1)
        #smooth_val, smooth_rev_val = calc_feat_smooth(adj, feat)
        feat_smooth_matrix = calc_feat_smooth(adj, feat)
        #print(smooth_val, smooth_rev_val.item())
        features = feat
        
        nx_g = nx.Graph(adj+ sp.eye(adj.shape[0]))
        
        g = dgl.from_networkx(nx_g)
    else:
        raise ValueError("wrong dataset name")
    
    max_train = 20
    
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    
    labels = torch.LongTensor(labels)
    #idx_train = torch.LongTensor(idx_train)
    #idx_val = torch.LongTensor(idx_val)
    #idx_test = torch.LongTensor(idx_test)
    
    #if len(new_classes) > 0:
    #    nb_classes = max(labels[idx_val]).item()
    #else:
    nb_classes = max(labels).item() + 1

    #
    xent = nn.CrossEntropyLoss(reduction='none')
    #xent = nn.CrossEntropyLoss()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    STAGE = 'pretrain'
    print('number of classes {}'.format(nb_classes))
    #output_edgelist(g, open('{}_edgelist.txt'.format(args.dataset), 'w'))
    #pos_emb = read_posit_emb(open('{}_dw.emb'.format(args.dataset), 'r'))
    if torch.cuda.is_available():
        print('Using CUDA')
        g = g.to(device)
        features = features.cuda()
        labels = labels.cuda()
        #idx_train = idx_train.cuda()
        #idx_val = idx_val.cuda()
        #idx_test = idx_test.cuda()
        if args.dataset == 'ppi':
            test_labels = test_labels.cuda()
            val_labels = val_labels.cuda()
            val_features = val_features.cuda()
            test_features = test_features.cuda()

    

    # inductive setting
    if False:
        if args.dataset == 'ppi':
            val_lbls = val_labels[idx_val]
            test_lbls = test_labels[idx_test]
        else:
            val_lbls = labels[idx_val]
            test_lbls = labels[idx_test]
    
    best_val_acc = 0
    cnt_wait = 0
    finetune = False
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []
    
    #print("original length:{}".format(len(ori_idx_train)))
    
    num_seeds = []
    all_runs_data = defaultdict(list)
    feature_smoothness = []
    embedding_smoothness = []
    avg_dist, max_dist = [], []
    #pre-compute stage
    #if args.biased_sample:
    if False:
        ppr_vector = torch.FloatTensor(calc_ppr_exact(adj, 0.01))
        ppr_dist = pairwise_distances(ppr_vector)
        
        #feat_dist = pairwise_distances(torch.FloatTensor(feat))
        pickle.dump({'ppr_vector':ppr_vector, 'ppr_dist': ppr_dist}, open('intermediate/{}_dump.p'.format(args.dataset), 'wb'))
    #else:
    else:
        train_dump = pickle.load(open('intermediate/{}_dump.p'.format(args.dataset), 'rb'))
        ppr_vector = train_dump['ppr_vector']
        ppr_dist = train_dump['ppr_dist']
        #Z = ppr_vector.matmul(feat)
    embed()
    #
    avg_mmd_dist = []
    training_seeds_run = pickle.load(open('data/localized_seeds_{}.p'.format(args.dataset), 'rb'))
    #assert len(training_seeds_run) == args.n_repeats
    for _run in range(args.n_repeats):
        # biased training data
        if args.biased_sample:
            # generate biased sample
            if False:
                train_seeds, _, _, _, _, _ = utils.createDBLPTraining(one_hot_labels, ori_idx_train, idx_val, idx_test, max_train = 1, new_classes=new_classes, unknown=unk)
                
                label_idx = []
                if args.dataset == 'pubmed':
                    num_pool = 10000
                elif args.dataset == 'cora':
                    num_pool = 1500
                else:
                    num_pool = 1000
                for i in train_seeds:
                    label_idx.append(torch.where(labels[:num_pool] == labels[i])[0])
                ppr_init = {}
                for i in train_seeds:
                    ppr_init[i] = 1
                print(train_seeds)

                idx_train = []
                for idx in range(len(train_seeds)):
                    idx_train += label_idx[idx][ppr_dist[train_seeds[idx], label_idx[idx]].argsort()[:max_train]].tolist()
                    #idx_train += label_idx[idx][ppr_vector[train_seeds[idx], label_idx[idx]].argsort()].tolist()[::-1][:max_train]
                training_seeds_run.append(idx_train)
            # use dumped biased sample 
            else:
                idx_train = training_seeds_run[_run]
            label_balance_constraints = np.zeros((labels.max().item()+1, len(idx_train)))
            for i, idx in enumerate(idx_train):
                label_balance_constraints[labels[idx], i] = 1
            #embed()

            all_idx = set(range(g.number_of_nodes())) - set(idx_train)
            #print(len(all_idx), g.number_of_nodes())
            if args.dataset == 'cora':
                idx_test = list(all_idx)
            iid_train, _, _, _, _, _ = utils.createDBLPTraining(one_hot_labels, ori_idx_train, idx_val, idx_test, max_train = max_train)
            #embed()
            #iid_train = np.random.choice(np.arrange(g.number_of_nodes()), max_train, replace=False).tolist()
            if args.arch >= 3:
                kmm_weight, MMD_dist = KMM(ppr_vector[idx_train, :], ppr_vector[iid_train, :], label_balance_constraints)

        else:
            #if args.dataset != 'ogbn-arxiv':
            idx_seed = np.random.randint(0,features.shape[0])
            idx_train, _, _, _, _, _ = utils.createDBLPTraining(one_hot_labels, ori_idx_train, idx_val, idx_test, max_train = max_train, new_classes=new_classes, unknown=unk)
            #embed()
            all_idx = set(range(g.number_of_nodes())) - set(idx_train)
            label_balance_constraints = np.zeros((labels.max().item()+1, len(idx_train)))
            for i, idx in enumerate(idx_train):
                label_balance_constraints[labels[idx], i] = 1
            # kmm_weight, MMD_dist = KMM(ppr_vector[idx_train, :], ppr_vector[idx_test, :], label_balance_constraints)
            
            test_lbls = labels[idx_test]

                
        train_lbls = labels[idx_train]
        reg_lbls = torch.cat([torch.ones(len(idx_train), dtype=torch.long), torch.zeros(len(idx_train), dtype=torch.long)]).cuda()
        # print(Counter(train_lbls.cpu().detach().numpy().tolist()))
        # enlarge the validation pool for DBLP 
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
                    F.tanh,
                    args.dropout,
                    args.aggregator_type
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
                    args.aggregator_type
                    )
        else:
            #model = GCN(ft_size, args.n_hidden, nb_classes, args.n_layers, F.relu, args.dropout, False)
            
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

        model.cuda()
        #train_loader = DataLoader(idx_train, batch_size = 200, shuffle=True)
        best_acc, best_epoch = 0.0, 0.0
        #torch.autograd.set_detect_anomaly(True)
        plot_x, plot_y, plot_z = [], [], []
        for epoch in range(args.n_epochs):
            if args.arch == 4 and epoch % 20 == 1:
            #    Z = ppr_vector.matmul(model.h.detach().cpu())
                kmm_weight, MMD_dist = KMM(model.h[idx_train, :].detach().cpu(), model.h[idx_test, :].detach().cpu(), label_balance_constraints)
            #else:
            #    kmm_weight = None
            #np.random.shuffle(kmm_weight)
            if args.biased_sample and False:
                reg_samples = np.random.choice(list(all_idx), max_train * num_class).tolist()

            model.train()
            optimiser.zero_grad()
            #embed()
            
            
            
            #print(len(idx_train))
            #for train_batch in train_loader:
            #   
            if args.biased_sample and False:
                reg_logits = model.reg_output(features)
                loss_1, loss_reg = xent(logits[idx_train], train_lbls), 0.2 * xent(reg_logits[idx_train+reg_samples], reg_lbls)
                loss =  loss_1 # + loss_reg
                #loss =  loss_1
                # print(loss_1.item(), loss_reg.item()) 
            else:
                if args.dataset != 'ogbn-arxiv':
                    logits = model(features)
                    loss = xent(logits[idx_train], labels[idx_train])
                else:
                    logits = model(features)
                    loss = cross_entropy(logits[idx_train], labels[idx_train])
                
                if args.arch == 0:
                    loss = loss.mean()
                    total_loss = loss
                elif args.arch == 1:
                    loss = loss.mean()
                    #total_loss = loss
                    #total_loss = loss + 1 * cmd(model.h[idx_train, :], model.h[idx_test, :])
                    #total_loss = loss + 1 * MMD(logits[idx_train, :], logits[idx_test, :])
                    total_loss = loss + 1 * MMD(model.h[idx_train, :], model.h[idx_test, :])
                elif args.arch == 2:
                    loss = loss.mean()
                    total_loss = loss + 1 * cmd(model.h[idx_train, :], model.h[iid_train, :])
                elif args.arch in [3,4]:
                    loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean()
                    #total_loss = loss
                    total_loss = loss +  1 * cmd(model.h[idx_train, :], model.h[iid_train, :])
                elif args.arch == 5:
                    loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean()
                    total_loss = loss
                    #total_loss = loss + 1 * MMD(logits[idx_train, :], logits[idx_test, :])
                #
            # preds = torch.argmax(logits[idx_train], dim=1).detach()
            if False and epoch % 1 == 0:
                #print(epoch, loss.item(), cmd(model.h[idx_train, :], model.h[idx_test, :]).item())
                plot_x.append(epoch)
                plot_y.append(loss.item())
                #plot_z.append(cmd(logits[idx_train, :], logits[idx_test, :]).item())
                plot_z.append(cmd(model.h[idx_train, :], model.h[idx_test, :]).item())
            if False and epoch % 50 == 0:
                #cmd
                #pass
                #
                print("current MMD is {}".format(MMD(logits[idx_train, :], logits[idx_test, :]).detach().cpu().item()))
                print("current CMD is {}".format(cmd(model.h[idx_train, :], model.h[idx_test, :]).detach().cpu().item()))
            total_loss.backward()
            optimiser.step()
            with torch.no_grad():
                if epoch % 10 == 0 and args.dataset == 'ogbn-arxiv':
                
                    model.eval()
                    #logits = model(features, bns=True)
                    logits = model(features)
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds[idx_train] == train_lbls.view(-1)).sum().float().item() / preds[idx_train].shape[0]
                    #val_acc = (preds[idx_val] == labels[idx_val].view(-1)).sum().float().item() / preds[idx_val].shape[0]
                    #test_acc = (preds[idx_test] == labels[idx_test].view(-1)).sum().float().item() / preds[idx_test].shape[0]
                    val_acc = compute_acc(logits[idx_val], labels[idx_val], evaluator)
                    test_acc = compute_acc(logits[idx_test], labels[idx_test], evaluator)
                    cmd_test = cmd(model.h[idx_train, :], model.h[idx_test, :]).item()
                    print("epoch:{}, loss:{}, cmd:{}, train acc:{}, valid acc:{}, test acc:{} ".format(epoch, loss.item(), cmd_test, acc, val_acc, test_acc))
                if False and epoch % 50 == 0:
                    model.eval()
                    logits = model(features)
                    preds_all = torch.argmax(logits, dim=1)
                    acc_val = f1_score(labels[idx_val].cpu(), preds_all[idx_val].cpu(), average='micro')
                    print(epoch, total_loss.item(), loss.item(), acc_val)
                    if acc_val > best_acc:
                        best_acc = acc_val
                        best_epoch = epoch
                        torch.save(model.state_dict(), 'best_model_{}.pt'.format(args.dataset))
        #print("best epoch:{}, best validation acc:{}".format(best_epoch, best_acc))
        model.eval()
        embeds = model(features).detach()
        
            # preds = nb_classes - F.softmax(logits, dim=1).max(dim=1)[0].gt(t).long() * (logits.argmax(dim=1)+1)
        # embed()
        logits = embeds[idx_test]
        #embeds = model(features).cpu().detach()
        #embeds = M.matmul(model(features)).detach().cpu()
        preds_all = torch.argmax(embeds, dim=1)
        embeds = embeds.cpu()
        #kmm_weight, MMD_dist = KMM(embeds[idx_train, :], embeds[idx_test, :], label_balance_constraints)
        #kmm_weight = naiveIW(ppr_vector[idx_train, :], ppr_vector[idx_test, :], label_balance_constraints)
        
        ## plot part
        #
        
        #plt.plot(plot_x, plot_y)
        #plt.plot(plot_x, plot_z)
        #plt.savefig('new_uniform.png')
        #print(plot_y[-1], plot_z[-1])
        
        
        #print("After", kmm_weight.max().item(), kmm_weight.min().item(), MMD_dist)
        #
        #embed()
        if False:
            min_max_scaler = preprocessing.MinMaxScaler()
            emb = min_max_scaler.fit_transform(embeds.cpu().numpy())
        else:
            emb = embeds.cpu().numpy()
            #emb = pos_emb.numpy()
        #embed()

        
        #embedding_smoothness.append(calc_emb_smooth(adj, emb))
        #tmp = pairwise_distances(emb[idx_train, :])
        #tmp = pairwise_distances(torch.FloatTensor(emb))
        #embed()
        #optional 
        if False:
            #embed()
            path_seeds, path_cnt, path_correct = [], defaultdict(int),defaultdict(int)
            for _idx in idx_train:
                path_seeds.append(nx.shortest_path(nx_g, source=_idx))
            
            for _idx in idx_test:
                min_dst = []
                for p in path_seeds:
                    if _idx in p:
                        min_dst.append(len(p[_idx])-1)
                if len(min_dst) == 0:
                    continue
                min_dst = min(min_dst)
                path_cnt[min(min_dst,4)] += 1
                if preds_all[_idx] == labels[_idx]:
                    path_correct[min(min_dst,4)] += 1
            for k in path_cnt:
            #    print(k, path_correct[k] / path_cnt[k], path_cnt[k])
                all_runs_data[k].append(path_correct[k] / path_cnt[k])
            
            if args.biased_sample:
                pickle.dump(all_runs_data, open('snowball_acc_distance_{}.p'.format(args.dataset), 'wb'))
            else:
                pickle.dump(all_runs_data, open('normal_acc_distance_{}.p'.format(args.dataset), 'wb'))
            for k in all_runs_data:
                print(k, np.mean(all_runs_data[k]))
        micro_f1.append(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro'))
        macro_f1.append(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='macro'))
    
    return micro_f1, macro_f1, avg_mmd_dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR-GNN')
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
    parser.add_argument("--verbose", type=bool, default=False,
                        help="print verbose step-wise information")
    parser.add_argument("--n-repeats", type=int, default=20,
                        help=".")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--dataset',type=str, default='cora')
    parser.add_argument('--num-unseen',type=int, default=1)
    parser.add_argument('--metapaths', type=list, default=['PAP'])
    parser.add_argument('--new-classes', type=list, default=[])
    parser.add_argument('--sc', type=float, default=0.0, help='GCN self connection')
    args = parser.parse_args()
    #
    #
    #print('here')
    torch.manual_seed(2)
    np.random.seed(11)
    if args.dataset == 'cora':
        num_class = 7
    elif args.dataset == 'citeseer':
        num_class = 6
    elif args.dataset == 'ppi':
        num_class = 9
    elif args.dataset == 'dblp':
        num_class = 5
    # 3 both techniques, 2 regularization only, 0 vanilla model
    
    if args.SR and args.gnn_arch == 'ppnp':
        args.arch = 3
    elif args.SR:
    #if args.SR:
        args.arch = 2
    else:
        args.arch = 0
#print(args)
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []
    #for i in utils.generateUnseen(num_class, args.num_unseen):
    micro_f1, macro_f1, out_acc = main(args, [])
    torch.cuda.empty_cache()
    # embed()
    #print(np.mean(in_acc), np.std(in_acc), np.mean(out_acc), np.std(out_acc))
    print("arch {}:".format(args.gnn_arch), np.mean(micro_f1), np.std(micro_f1), np.mean(macro_f1), np.std(macro_f1))
        #print(out_acc)
        #plt.scatter(out_acc, micro_f1)
        #plt.scatter(X_embedded[idx_train, 0], X_embedded[idx_train, 1], 10 * kmm_weight)
        #plt.savefig('{}_{}_cmd.png'.format(args.dataset, args.gnn_arch))
        #break
