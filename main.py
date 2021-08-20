
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
from utils import *
from log_helper import *

def get_batch_adj(batch_idx, adj):
    densed_batch_adj = torch.Tensor()
    for idx in batch_idx:
        densed_batch_adj = torch.cat([densed_batch_adj, adj[idx].to_dense()])
    return densed_batch_adj

if __name__=='__main__':
    
    data = Data('./dataset/oag')
    
    with open('data2.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    '''
    
    with open('data2.pkl', 'rb') as f:
        data = pickle.load(f)
    '''
    model = GCN(nfeat = data.features.shape[1], nhid=160, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    log_save_id = create_log_id('./log/')
    logging_config(folder='./log/', name='log{:d}'.format(log_save_id), no_console=False)
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    batch_size = 1024
    n_batch = data.n_node // batch_size + 1
    if torch.cuda.is_available():
        model.cuda()
        features = data.features.cuda()
        fadj = data.fadj.cuda()
    
    for epoch in range(1, 10001):
        total_loss = 0.
        start = time.time()
        for idx in range(n_batch):
            start_ = time.time()
            head, pos_tail, neg_tail = generate_batch(data.edge_total_dict, batch_size)
            model.train()
            optimizer.zero_grad()
            
            loss = model(features, fadj, head, pos_tail, neg_tail)
  
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / n_batch
            # if idx % 1000 == 0:
            #     print(epoch, idx, loss.item(), time.time() - start_)
        
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, './pt/cf_loss_' + str(epoch) + '_' + str(round(total_loss, 2)) + '.pt')
        
        logging.info(str(epoch) + ' | ' + str(round(total_loss, 2)) + ' | ' + str(round(time.time() - start, 2)))

        if epoch % 10000000 == 0:
            print('{:2d} {:.4f} {:.2f}s\n'.format(epoch, loss.item(), time.time() - start), end='\t')

            with torch.no_grad():
                model.eval()
                output = model(data.features, data.nfadj)
                #TBD