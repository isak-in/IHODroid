import random

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import classification_report
import sklearn

def load_data(ratio_generated, path="../dataset/citeseer/", dataset="citeseer"):
    dataset_path = 'E:/ImGAGN-main/ImGAGN-main/dataset/drebin/dataset_imb.npy'
    dataset = np.load(dataset_path, allow_pickle=True).item()
    permssion_mtx = dataset.get('permssion_mtx')
    intent_mtx = dataset.get('intent_mtx')
    api_call_mtx = dataset.get('api_call_mtx')
    activity_mtx = dataset.get('activity_mtx')
    real_permission_mtx = dataset.get('real_permission_mtx')
    provider_mtx = dataset.get('provider_mtx')
    url_mtx = dataset.get('url_mtx')
    call_mtx = dataset.get('call_mtx')
    service_mtx = dataset.get('service_mtx')

    labels = dataset.get('label')
    features = sp.csr_matrix(provider_mtx, dtype=np.float32)
    indices = np.arange(0, 100000)
    random.shuffle(indices)
    idx_train, idx_test = data_split(indices, 0.7)

    majority = np.array([x for x in indices if labels[x] == 1])
    minority = np.array([x for x in indices if labels[x] == 0])

    num_minority = minority.shape[0]
    num_majority = majority.shape[0]
    print("Number of majority: ", num_majority)
    print("Number of minority: ", num_minority)

    generate_node = []
    generate_label=[]
    for i in range(labels.shape[0], labels.shape[0]+int(ratio_generated*num_majority)-num_minority):
        generate_node.append(i)
        generate_label.append(1)
    idx_train = np.hstack((idx_train, np.array(generate_node)))
    print(idx_train.shape)

    minority_test = np.array([x for x in idx_test if labels[x] == 1])
    minority_all = np.hstack((minority, minority_test))


    labels= np.hstack((labels, np.array(generate_label)))


    edges = np.pad(np.int32(np.matmul(service_mtx, service_mtx.T) > 0), ((0, len(generate_label)), (0, len(generate_label))))

    adj_real = sp.coo_matrix(edges, shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)


    adj = adj_real + adj_real.T.multiply(adj_real.T > adj_real) - adj_real.multiply(adj_real.T > adj_real)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    generate_node=torch.LongTensor(np.array(generate_node))
    minority = torch.LongTensor(minority)
    majority = torch.LongTensor(majority)
    minority_all = torch.LongTensor(minority_all)

    return adj, adj_real, features, labels, idx_train, idx_test, generate_node, minority, majority, minority_all




def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels, output_AUC):
    preds = output.max(1)[1].type_as(labels)


    recall = sklearn.metrics.recall_score(labels.cpu().numpy(), preds.cpu().numpy())
    f1_score = sklearn.metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy())
    AUC = sklearn.metrics.roc_auc_score(labels.cpu().numpy(), output_AUC.detach().cpu().numpy())
    acc = sklearn.metrics.accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    precision = sklearn.metrics.precision_score(labels.cpu().numpy(), preds.cpu().numpy())
    return recall, f1_score, AUC, acc, precision


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def add_edges(adj_real, adj_new):
    adj = adj_real+adj_new
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def print_edges_num(dense_adj, labels):
    c_num = labels.max().item()+1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            #ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            print("edges between class {:d} and class {:d}: {:f}".format(i,j,edge_num))


def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    return np.int64(full_list[:offset]), np.int64(full_list[offset:])



if __name__ == '__main__':
    load_data(1, path="../dataset/cora/", dataset="cora")