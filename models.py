#cf loss with https://github.com/LunaBlack/KGAT-pytorch/blob/e7305c3e80fb15fa02b3ec3993ad3a169b34ce64/model/KGAT.py#L192
import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# The code of forwarding and testing is from https://github.com/LunaBlack/KGAT-pytorch
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
    
    def get_embedding(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x

    def forward(self, x, adj, head, pos_tail, neg_tail):
        x = self.get_embedding(x, adj)
        
        head_embed = x[head]
        pos_tail_embed = x[pos_tail]
        neg_tail_embed = x[neg_tail]
        
        #loss with kgat - cf loss 
        pos_score = torch.sum(head_embed * pos_tail_embed, dim=1)
        neg_score = torch.sum(head_embed * neg_tail_embed, dim=1)
        
        loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        loss = torch.mean(loss)
        
        l2_loss = self._L2_loss_mean(head_embed) + self._L2_loss_mean(pos_tail_embed) + self._L2_loss_mean(neg_tail_embed)
        loss = loss + 1e-5 * l2_loss
        
        return loss

    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)
    
    #https://github.com/DaehanKim/vgae_pytorch/blob/master/train.py
    def test(self, x, adj, pos_edges, neg_edges):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        x = self.get_embedding(x, adj)

        pos = []
        for e1, e2 in pos_edges:
            pos.append(sigmoid(np.dot(x[e1], x[e2])))

        neg = []
        for e1, e2 in neg_edges:
            neg.append(sigmoid(np.dot(x[e1], x[e2])))
        
        preds_all = np.hstack([pos, neg])
        labels_all = np.hstack([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

        # cf_score = torch.matmul(x[batch_idx], x.transpose(0, 1))
        # return cf_score