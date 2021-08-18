
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
    
    data = Data('./dataset/oag')
    
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    '''
    
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    '''
    import pdb; pdb.set_trace()
    

    model = GCN(nfeat = data.features.shape[1], nhid=160, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    adj = data.fadj
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    batch_size = 32
    n_batch = data.n_node // batch_size + 1
    if torch.cuda.is_available():
        model.cuda()
        features = data.features.cuda()
        fadj = data.fadj.cuda()
    
    for epoch in range(0, 10001):
        total_loss = 0.
        start = time.time()
        for idx in range(0, data.n_node, batch_size):
            start_ = time.time()
            idx_end = idx + batch_size
            if idx_end > data.n_node:
                idx_end = data.n_node
            batch_idx = [i for i in range(idx, idx_end)]
            
            model.train()
            optimizer.zero_grad()
            adj_pred = model(features, fadj, batch_idx)

            loss = F.binary_cross_entropy(adj_pred.view(-1), get_batch_adj(batch_idx, adj).cuda())
            # loss = norm * F.binary_corss_entropy(adj_pred.view(-1), adj.to_dense().view(-1))
            # loss_train = F.nll_loss(output[data.idx_train], lab[data.idx_train])

                
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / n_batch
            if idx % 100 == 0:
                print(epoch, idx, loss.item(), time.time() - start_, adj_pred.view(-1))

        if epoch % 1 == 0:
            
            print('{:2d} {:.4f} {:.2f}s\n'.format(epoch, loss.item(), time.time() - start), end='\t')

            with torch.no_grad():
                model.eval()
                output = model(data.features, data.nfadj)
                #TBD