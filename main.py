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

if __name__=='__main__':
    
    data = Data('./dataset/oag')
    
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    '''
    
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    '''
        
    model = GCN(nfeat = data.features.shape[1], nhid=160, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    adj = data.fadj
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    
    for epoch in range(0, 10001):
        start = time.time()
        model.train()
        optimizer.zero_grad()
        adj_pred = model(data.features, data.fadj)

        loss = norm * F.binary_corss_entropy(adj_pred.view(-1), adj.to_dense().view(-1))
        #loss_train = F.nll_loss(output[data.idx_train], lab[data.idx_train])
            
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimzier_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, './pt/test.pt')
            print('{:2d} {:.4f} {:.2f}s\n'.format(epoch, loss.item(), time.time() - start), end='\t')

            with torch.no_grad():
                model.eval()
                output = model(data.features, data.nfadj)
                #TBD