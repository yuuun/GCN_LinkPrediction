# GCN_LinkPrediction

This is the code of GCN with the task of link prediction.

The dataset I use was 'OAG(Open Acamedic Graph)' which was preprocessed from the GPT-GNN.



Since the size of the dataset was huge(number of node: 19M), it was impossible to use binary cross classification loss.

Here, I used bpr loss which increases (positive score - negative score) used from the kgat(Knowledge Graph Attention Network)
