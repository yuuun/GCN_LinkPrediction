
from collections import defaultdict
import random as rd
import numpy as np
import scipy.sparse as sp
import torch
import os

class Data():
    def __init__(self, data_path):
        self.data_path = data_path
        edge_path = data_path + '.edge'
        feature_path = data_path + ".feature"

        self.load_feature(feature_path)
        self.load_edge(edge_path)
        self.fadj = self.load_adj()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx  
        
    def load_adj(self):
        edges = np.array(self.edge_list, dtype=np.int32)
        fadj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(self.n_node, self.n_node), dtype=np.float32)
        fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
        
        fadj = self.normalize(fadj + sp.eye(fadj.shape[0]))
        fadj = self.sparse_mx_to_torch_sparse_tensor(fadj)

        return fadj 
    
    def load_edge(self, edge_path):
        lines = open(edge_path, 'r').readlines()
        edge_dict = dict()
        self.edge_total_dict = defaultdict(list)
        self.edge_list = []
        self.total_edge_list = []
        
        for l in lines:
            tmp = l.strip() #deleting '\n'
            val = [int(i) for i in tmp.split()]
            node_id = val[0] 
            linked_node = val[1:]
        
            edge_dict[node_id] = linked_node
            
            for node in linked_node:
                self.edge_total_dict[node_id].append(node)
                self.edge_total_dict[node].append(node_id)
            for ln in linked_node:
                self.edge_list.append([node_id, ln]) 
                self.total_edge_list.append([node_id, ln])
                self.total_edge_list.append([ln, node_id])

        if not(os.path.isfile('./dataset/test.edge') and os.path.isfile('./dataset/train.edge')):
            self.train_list, self.test_list = self.sample_edge()
        else:
            self.train_list = self.load_sampled_edge('./dataset/train.edge')
            self.test_list = self.load_sampled_edge('./dataset/test.edge')

        self.train_adj = self.load_adj()
        
        if os.path.isfile('./data/false.test'):
            self.load_false_edge()
        else:
            self.make_test_edge()

    def make_test_edge(self):
        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)
            
        false_test = []
        while len(false_test) < len(self.test_list):
            idx_i = np.random.randint(0, self.n_node)
            idx_j = np.random.randint(0, self.n_node)
            if idx_i == idx_j:
                continue
            if idx_i in self.edge_total_dict[idx_j] or idx_j in self.edge_total_dict[idx_i]:
                continue
            if false_test:
                if ismember([idx_j, idx_i], np.array(false_test)):
                    continue
                if ismember([idx_i, idx_j], np.array(false_test)):
                    continue
            false_test.append([idx_i, idx_j])
        
        self.false_test = false_test
        with open('./dataset/false.test', 'w') as f:
            for e1, e2 in false_test:
                f.write(str(e1) + ' ' + str(e2) + '\n')

    def load_false_edge(self):
        lines = open('./dataset/false.test', 'r').readlines()
        self.false_test = []
        for l in lines:
            val = [int(i) for i in l.strip().split()]
            self.false_test.append([val[0], val[1]])

    def sample_edge(self):
        exclude_edge = []

        for node, linked_node in self.edge_total_dict.items():
            if len(linked_node) < 3:
                exclude_edge.append(node)

        candidate_edge_list = []
        train_list = []
        for e1, e2 in self.edge_list:
            if e1 not in exclude_edge and e2 not in exclude_edge:
                candidate_edge_list.append([e1, e2])
            else:
                train_list.append([e1, e2])

        n_test = int(len(self.edge_list) * 0.1)
        rd.shuffle(candidate_edge_list)
        test_list = sorted(candidate_edge_list[:n_test], key=lambda x:(x[0], x[1]))
        
        train_list = train_list + candidate_edge_list[n_test:]
        train_list = sorted(train_list, key=lambda x:(x[0], x[1]))
        
        with open('./dataset/train.edge', 'w') as f:
            for e1, e2 in train_list:
                f.write(str(e1) + ' ' + str(e2) + '\n')

        with open('./dataset/test.edge', 'w') as f:
            for e1, e2 in test_list:
                f.write(str(e1) + ' ' + str(e2) + '\n')
        
        return train_list, test_list

    def load_sampled_edge(self, data_path):
        lines = open(data_path, 'r').readlines()
        edge_list = []
        for l in lines:
            val = [int(i) for i in l.strip().split()]
            edge_list.append([val[0], val[1]])
        return edge_list

    def load_feature(self, feature_path):
        lines = open(feature_path, 'r').readlines()
        feature_list = []
        for l in lines:
            tmp = l.strip()
            val = [float(i) for i in tmp.split()]
            feature_list.append(val[1:])

        self.n_node = len(feature_list)
        self.features = sp.csr_matrix(np.array(feature_list), dtype=np.float32)
        self.features = torch.FloatTensor(np.array(self.features.todense()))