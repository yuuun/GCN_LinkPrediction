import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
        self.loss = nn.BCELoss()
        
    def forward(self, x, adj, train_idx):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        adj_ = x @ x.T
        out = self.loss(adj[train_idx], adj_[train_idx])

        return out
        
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.gc2(x, adj)
               
        #return F.log_softmax(x, dim=1)
