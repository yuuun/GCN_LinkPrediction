import random as rd
import torch

def generate_batch(edge_total_dict, batch_size):
    n_node = len(edge_total_dict)
    head = [rd.randint(0, n_node) for _ in range(batch_size)]

    pos_tail, neg_tail = [], []
    for h in head:
        pos_tail.append(rd.sample(edge_total_dict[h], 1)[0])
        neg_tail.append(rd.choice([i for i in range(0, n_node) if i not in edge_total_dict[h]]))
    
    head = torch.LongTensor(head)
    pos_tail = torch.LongTensor(pos_tail)
    neg_tail = torch.LongTensor(neg_tail)

    return head, pos_tail, neg_tail