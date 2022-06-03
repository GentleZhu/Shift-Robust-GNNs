import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv,GraphConv,GATConv, SGConv
import scipy.sparse as sp
import numpy as np
import math
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def norm2(samples):
    return F.normalize(samples, p=2, dim=1)

def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)

class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)

class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)

def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr

#def calc_

class PPRPowerIteration(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)
        self.fcs = nn.ModuleList([nn.Linear(in_feats, n_hidden, bias=False), nn.Linear(n_hidden, n_classes, bias=False)])
        self.disc = nn.Linear(n_hidden, 2)
        self.bns = nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
    def forward(self, local_preds: torch.FloatTensor, bns = False):
        
        for l_id, layer in enumerate(self.fcs):
            local_preds = self.dropout(local_preds)
            if l_id != len(self.fcs) - 1:
                #print('here')
                local_preds = layer(local_preds)
                if bns:
                    local_preds = self.bns[l_id](local_preds)
                local_preds = F.tanh(local_preds)
            else:
                self.h = local_preds
                local_preds = layer(local_preds)
        
        preds = local_preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds
        return preds
    
    def reg_output(self, idx_train, alpha=1):
        reverse_feature = ReverseLayerF.apply(self.h[idx_train,:], alpha)
        #reverse_feature = self.h
        return self.disc(reverse_feature)
        #return self.fcs[1](F.relu(self.fcs[0](reverse_feature)))

class GraphSAGE(nn.Module):
    # change to the form of final layer to be linear
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        self.activation = activation
        # input layer
        #self.norm = torch.norm(p=None, )
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, norm=norm2, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers-1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, norm=norm2, feat_drop=dropout, activation=activation))
        # output layer
        #self.layers.append(nn.Linear(n_hidden, n_classes))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None
    '''
    def forward(self, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        self.final_hidden = h
        return self.layers[-1](h)
    '''
    def forward(self, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        self.h = h
        return self.layers[-1](h)
        #return h
    def output(self, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        return h

class Net(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=None))
        #self.fcs.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        self.activation = activation
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        for i in range(n_layers-1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=None))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=None)) # activation None
        #self.layers.append(nn.Linear(n_hidden, n_classes))
        self.fcs = nn.ModuleList([nn.Linear(n_hidden, n_hidden, bias=True), nn.Linear(n_hidden, 2, bias=True)])
        self.disc = GraphConv(n_hidden, 2, activation=None)
        self.dropout = nn.Dropout(p=dropout)
    '''
    def forward(self, features):
        h = features
        for l_id, layer in enumerate(self.fcs):
            h = self.dropout(h)
            if l_id != len(self.fcs) - 1:
                #print('here')
                h = F.relu(layer(h))
            else:
                h = layer(h)
        #self.final_hidden = h
        return h
        #return self.layers[-1](self.g, h)
    '''
    
    def forward(self, features, bns=False):
        h = features
        for idx, layer in enumerate(self.layers[:-1]):
            h = layer(self.g, h)
            if bns:
                h = self.bns[idx](h)
            h = self.activation(h)
            h = self.dropout(h)
            #if idx == 0:
            #    self.h = h
            #if idx == len(self.layers) - 2:
        self.h = h
        
        #for layer in self.fcs:
        #    h = layer(h)
        #return h
        #h = self.dropout(h)
        return self.layers[-1](self.g, h)
    
    def reg_output(self, idx_train, alpha=1):
        reverse_feature = ReverseLayerF.apply(self.h, alpha)
        #reverse_feature = self.h
        return self.disc(self.g, reverse_feature)[idx_train,:]
        #return self.fcs[1](F.relu(self.fcs[0](reverse_feature)))

    def output(self, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        return h

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 num_heads=8):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(GATConv(in_feats, n_hidden, num_heads=num_heads, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GATConv(n_hidden*num_heads, n_hidden, num_heads=num_heads, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(GATConv(n_hidden*num_heads, n_classes, num_heads=1, feat_drop=dropout, activation=None)) # activation None


        #embed()
    def forward(self, features, bns=False):
        h = features
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](self.g, h).flatten(1)
        self.h = h
        return self.layers[-1](self.g, h).mean(1)

    def output(self, g, features):
        h = features
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](g, h).flatten(1)
        return self.layers[-1](g, h).mean(1)


class SGC(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 train_mask):
        super(SGC, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SGConv(in_feats, n_hidden, k=2, cached=True))
        self.linear = nn.Linear(n_hidden, n_classes)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.train_mask = train_mask
    def forward(self, features, bns=False):
        #if self.training:
        if False:
            h = features
            h[~self.train_mask] = 0
        else:
            h = features
        for layer in self.layers:
            h = layer(self.g, h)
            #h = self.dropout(h)
        #return h
        h = self.activation(h)
        self.h = h
        return self.linear(h)

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(DGI, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2


class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        #return torch.log_softmax(features, dim=-1)
        return features
