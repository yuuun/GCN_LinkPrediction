import random as rd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

def generate_batch(edge_total_dict, batch_size):
    n_node = len(edge_total_dict)
    head = rd.sample([*edge_total_dict], batch_size)
    pos_tail, neg_tail = [], []
    for h in head:
        head_list = edge_total_dict[h]
        pos_tail.append(rd.sample(head_list, 1)[0])
        
        while True:
            cand = rd.randint(0, n_node - 1)
            if cand not in head_list:
                neg_tail.append(cand)
                break
    
    head = torch.LongTensor(head)
    pos_tail = torch.LongTensor(pos_tail)
    neg_tail = torch.LongTensor(neg_tail)

    return head, pos_tail, neg_tail

### Evaulation Protocol: NDCG and Recall
### This is the code from https://github.com/LunaBlack/KGAT-pytorch/utils/metrics.py
### Since the dataset is too large, we cannot reconstruct the adjacency matrix

def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):
    """
    cf_scores: (n_users, n_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['recall']    = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]['ndcg']      = ndcg_at_k_batch(binary_hit, k)
    return metrics_dict