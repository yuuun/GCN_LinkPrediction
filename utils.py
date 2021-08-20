import random as rd
import torch

def generate_batch(edge_total_dict, batch_size):
    n_node = len(edge_total_dict)
    head = [rd.randint(0, n_node - 1) for _ in range(batch_size)]
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