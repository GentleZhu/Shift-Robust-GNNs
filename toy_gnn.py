from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
import utils
import numpy as np
import pickle
import networkx as nx
import scipy.sparse as sp
import dgl
from sklearn.metrics import f1_score

def KMM(X,Xtest,_A=None, _sigma=1e1,beta=0.2):

    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(- 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    H /= 3
    f /= 3
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()
    
    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0],1)))
    G = - np.eye(nsamples)
    _A = _A[~np.all(_A==0, axis=1)]
    b = _A.sum(1)
    h = - beta * np.ones((nsamples,1))
    
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    sol=solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h), matrix(_A), matrix(b))
    return np.array(sol['x']), MMD_dist.item()

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
    return torch.clamp(dist, 0.0, np.inf)

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

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class ToyGNN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(ToyGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        #print(in_feats, n_hidden, n_classes)
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=None))

        # hidden layers
        self.activation = activation
        for i in range(n_layers-1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=None))
        # output layer hidden units -> n_classes
        self.layers.append(GraphConv(n_hidden, n_classes, activation=None)) # activation None
        self.fcs = nn.ModuleList([nn.Linear(n_hidden, n_hidden, bias=True), nn.Linear(n_hidden, 2, bias=True)])
        self.disc = GraphConv(n_hidden, 2, activation=None)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for idx, layer in enumerate(self.layers[:-1]):
            h = layer(self.g, h)
            h = self.activation(h)
            h = self.dropout(h)
        self.h = h

        return self.layers[-1](self.g, h)
    
    def dann_output(self, idx_train, iid_train, alpha=1):
        reverse_feature = ReverseLayerF.apply(self.h, alpha)
        dann_loss = xent(self.disc(self.g, reverse_feature)[idx_train,:], torch.ones_like(labels[idx_train])).mean() + xent(self.disc(self.g, reverse_feature)[iid_train,:], torch.zeros_like(labels[iid_train])).mean()
        return dann_loss
    
    def shift_robust_output(self, idx_train, iid_train, alpha = 1):
        return alpha * cmd(self.h[idx_train, :], self.h[iid_train, :])

    def output(self, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        return h

if __name__ == '__main__':
    DATASET = 'citeseer'
    EPOCH = 200
    # option of 'SRGNN','DANN' and None
    METHOD = 'SRGNN'
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    adj, features, one_hot_labels, ori_idx_train, idx_val, idx_test = utils.load_data(DATASET)
    nx_g = nx.Graph(adj+ sp.eye(adj.shape[0]))
    g = dgl.from_networkx(nx_g).to(device)
    labels = torch.LongTensor([np.where(r==1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]).to(device)
    features = torch.FloatTensor(utils.preprocess_features(features)).to(device)
    xent = nn.CrossEntropyLoss(reduction='none')
    

    model = ToyGNN(g,features.shape[1],32,labels.max().item() + 1,1,F.tanh,0.2)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    model.cuda()
    # an example of biased training data
    idx_train = torch.LongTensor(pickle.load(open('data/localized_seeds_{}.p'.format(DATASET), 'rb'))[0])
    all_idx = set(range(g.number_of_nodes())) - set(idx_train)
    idx_test = torch.LongTensor(list(all_idx))
    perm = torch.randperm(idx_test.shape[0])
    iid_train = idx_test[perm[:idx_train.shape[0]]] 
    
    Z_train = torch.FloatTensor(adj[idx_train.tolist(), :].todense())
    Z_test = torch.FloatTensor(adj[iid_train.tolist(), :].todense())
    #embed()
    label_balance_constraints = np.zeros((labels.max().item()+1, len(idx_train)))
    for i, idx in enumerate(idx_train):
        label_balance_constraints[labels[idx], i] = 1
    #embed()
    kmm_weight, MMD_dist = KMM(Z_train, Z_test, label_balance_constraints, beta=0.2)
    print(kmm_weight.max(), kmm_weight.min())
    for epoch in range(EPOCH):
        model.train()
        optimiser.zero_grad()
        logits = model(features)
        loss = xent(logits[idx_train], labels[idx_train])
        if METHOD == 'SRGNN':
            #regularizer only: loss = loss.mean() + model.shift_robust_output(idx_train, iid_train)
            #instance-reweighting only: loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean()
            loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean() + model.shift_robust_output(idx_train, iid_train)
        elif METHOD == 'DANN':
            loss = loss.mean() + model.dann_output(idx_train, iid_train)
        elif METHOD is None:
            loss = loss.mean()
        loss.backward()
        optimiser.step()
    
    model.eval()
    embeds = model(features).detach()
    logits = embeds[idx_test]
    preds_all = torch.argmax(embeds, dim=1)

    print("Accuracy:{}".format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')))
