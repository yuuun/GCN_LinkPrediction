
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import math
import random
from sklearn.metrics import classification_report
from data_loader import Data
from models import *
import time
import pickle
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import scipy.sparse as sp

def get_batch_adj(batch_idx, adj):
    densed_batch_adj = torch.Tensor()
    for idx in batch_idx:
        densed_batch_adj = torch.cat([densed_batch_adj, adj[idx].to_dense()])
    return densed_batch_adj

if __name__=='__main__':
    '''
    data = Data('../dataset/oag')
    
    with open('../data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    '''
    
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    model = GCN(nfeat = data.features.shape[1], nhid=160, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    adj = data.fadj
    
    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    batch_size = 32
    n_batch = data.n_node // batch_size + 1
    if torch.cuda.is_available():
        model.cuda()
        features = data.features.cuda()
        fadj = data.fadj.cuda()
    
    for epoch in range(1, 10001):
        total_loss = 0.
        start = time.time()
        for idx in range(0, data.n_node, batch_size):
            end_idx = idx + batch_size
            if end_idx > data.n_node:
                end_idx = data.n_node
            batch_idx = [i for i in range(idx, end_idx)]

            start_ = time.time()
            model.train()
            optimizer.zero_grad()
            adj_pred = model(features, fadj, batch_idx)

            dd = time.time()
            loss = F.binary_cross_entropy(adj_pred.view(-1), get_batch_adj(batch_idx, adj))
  
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / n_batch
            if idx % 100 == 0:
                print(epoch, idx, loss.item(), time.time() - start_)
            
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, './pt/multiply_loss_' + str(epoch) + '.pt')
        
        if epoch % 1000 == 0:
            
            print('{:2d} {:.4f} {:.2f}s\n'.format(epoch, loss.item(), time.time() - start), end='\t')

            with torch.no_grad():
                model.eval()
                output = model(data.features, data.nfadj)
                #TBD



    def get_scores(edges_pos, edges_neg, adj_rec):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:

            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score
