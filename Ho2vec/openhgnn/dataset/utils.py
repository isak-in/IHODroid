import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch as th

import scipy
from scipy.sparse import csc_matrix


from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

"""
It's the dataset from HAN.
Refer to https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/utils.py
"""


def load_acm(remove_self_loop):
    url = 'dataset/ACM3025.pkl'
    data_path = './openhgnn/dataset/ACM3025.pkl'
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    labels, features = th.from_numpy(data['label'].todense()).long(), \
                       th.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data['PAP'])
    subject_g = dgl.from_scipy(data['PLP'])
    gs = [author_g, subject_g]

    train_idx = th.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = th.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = th.from_numpy(data['test_idx']).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask


def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    url = 'dataset/ACM.mat'
    data_path = './openhgnn/dataset/ACM.mat'
    if not os.path.exists(data_path):
        download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']  # paper-field?
    p_vs_a = data['PvsA']  # paper-author
    p_vs_t = data['PvsT']  # paper-term, bag of words
    p_vs_c = data['PvsC']  # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    # hg = dgl.heterograph({
    #     ('paper', 'pa', 'author'): p_vs_a.nonzero(),
    #     ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
    #     ('paper', 'pf', 'field'): p_vs_l.nonzero(),
    #     ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    # })

    # Graph(num_nodes={'author_src': 368, 'field_src': 23, 'paper_src': 3892, 'author_dst': 368, 'field_dst': 23, 'paper_dst': 128},
    #  num_edges={('author_src', 'ap', 'paper_dst'): 382, ('field_src', 'fp', 'paper_dst'): 128, ('paper_src', 'pa', 'author_dst'): 1717, ('paper_src', 'pf', 'field_dst'): 3861},
    #  metagraph=[('author_src', 'paper_dst', 'ap'), ('field_src', 'paper_dst', 'fp'), ('paper_src', 'author_dst', 'pa'), ('paper_src', 'field_dst', 'pf')])
    # block 修改
    # hg = dgl.heterograph({
    #     ('paper_src', 'pa', 'author_dst'): p_vs_a.nonzero(),
    #     ('author_src', 'ap', 'paper_dst'): p_vs_a.transpose().nonzero(),
    #     ('paper_src', 'pf', 'field_dst'): p_vs_l.nonzero(),
    #     ('field_src', 'fp', 'paper_dst'): p_vs_l.transpose().nonzero()
    # })

    # MAGNN处理
    hg = dgl.heterograph({
        ('paper', 'paper-author', 'author'): p_vs_a.nonzero(),
        ('author', 'author-paper', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'paper-field', 'field'): p_vs_l.nonzero(),
        ('field', 'field-paper', 'paper'): p_vs_l.transpose().nonzero()
    })
    features = th.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = th.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    hg.nodes['paper'].data['h'] = features
    # hg.nodes['paper'].data['emb'] = features
    hg.nodes['paper'].data['labels'] = labels
    hg.nodes['paper'].data['train_mask'] = train_mask
    hg.nodes['paper'].data['val_mask'] = val_mask
    hg.nodes['paper'].data['test_mask'] = test_mask

    return hg, 'paper', num_classes, features.shape[1]


def load_drebin_mal(remove_self_loop):
    assert not remove_self_loop
    # 数据集路径
    dataset_path = './openhgnn/dataset/drebin/dataset_6000.npy'

    dataset = np.load(dataset_path, allow_pickle=True).item()
    index = 0
    permssion_mtx = csc_matrix(dataset.get('permssion_mtx')[index:])
    intent_mtx = csc_matrix(dataset.get('intent_mtx')[index:])
    api_call_mtx = csc_matrix(dataset.get('api_call_mtx')[index:])
    activity_mtx = csc_matrix(dataset.get('activity_mtx')[index:])
    service_mtx = csc_matrix(dataset.get('service_mtx')[index:])
    # permission_type_mtx = csc_matrix(dataset.get('permission_type_mtx')[index:])
    # url_mtx = csc_matrix(dataset.get('url_mtx')[index:])
    # real_permission_mtx = csc_matrix(dataset.get('real_permission_mtx')[index:])
    # call_mtx = csc_matrix(dataset.get('call_mtx')[index:])
    # provider_mtx = csc_matrix(dataset.get('provider_mtx')[index:])

    data_label = dataset.get('label')[index:]

    hg = dgl.heterograph({
        # path1
        ('app', 'ap', 'permission'): vert(permssion_mtx),
        ('permission', 'pa', 'app'): vertT(permssion_mtx),
        # path2
        ('app', 'ain', 'intent'): vert(intent_mtx),
        ('intent', 'ina', 'app'): vertT(intent_mtx),
        # path3
        ('app', 'aap', 'api'): vert(api_call_mtx),
        ('api', 'apa', 'app'): vertT(api_call_mtx),
        # path4
        ('app', 'aat', 'activity'): vert(activity_mtx),
        ('activity', 'ata', 'app'): vertT(activity_mtx),
        # path5
        ('app', 'as', 'service'): vert(service_mtx),
        ('service', 'sa', 'app'): vertT(service_mtx),
        # path6
        # ('permission', 'ppt', 'permission_type'): vert(permission_type_mtx),
        # ('permission_type', 'ptp', 'permission'): vertT(permission_type_mtx),
        # # path7
        # ('app', 'atu', 'url'): vert(url_mtx),
        # ('url', 'uta', 'app'): vertT(url_mtx),
        # # path8 real_permission
        # ('app', 'atr', 'real_permission'): vert(real_permission_mtx),
        # ('real_permission', 'rta', 'app'): vertT(real_permission_mtx),
        # # path8 call_mtx
        # ('app', 'atc', 'call'): vert(call_mtx),
        # ('call', 'cta', 'app'): vertT(call_mtx),
        # path8 real_permission
        # ('app', 'atpp', 'provider'): vert(provider_mtx),
        # ('provider', 'ppta', 'app'): vertT(provider_mtx),
    })

    # 针对magnn修改
    # hg = dgl.heterograph({
    #     # path1
    #     ('app', 'app-permission', 'permission'): vert(permssion_mtx),
    #     ('permission', 'permission-app', 'app'): vertT(permssion_mtx),
    #     # path2
    #     ('app', 'app-intent', 'intent'): vert(intent_mtx),
    #     ('intent', 'intent-app', 'app'): vertT(intent_mtx),
    #     # path3
    #     ('app', 'app-api', 'api'): vert(api_call_mtx),
    #     ('api', 'api-app', 'app'): vertT(api_call_mtx),
    #     # path4
    #     ('app', 'app-activity', 'activity'): vert(activity_mtx),
    #     ('activity', 'activity-app', 'app'): vertT(activity_mtx),
    #     # path5
    #     ('app', 'app-service', 'service'): vert(service_mtx),
    #     ('service', 'service-app', 'app'): vertT(service_mtx),
    #     # path6
    #     # ('permission', 'permission-permission_type', 'permission_type'): vert(permission_type_mtx),
    #     # ('permission_type', 'permission_type-permission', 'permission'): vertT(permission_type_mtx),
    # })

    num_classes = 2

    indices = np.arange(0, 5000 - index)
    random.shuffle(indices)

    # 原始数据集
    train_idx, temp_idx = data_split(indices, 0.8)
    val_idx, test_idx = data_split(temp_idx, 0.5)

    # 生成节点集
    # train_idx, temp_idx = data_split_imb(indices)
    # val_idx, test_idx = data_split(temp_idx, 0.5)
    # ----------------------------



    num_nodes = hg.number_of_nodes('app')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    hg.nodes['app'].data['labels'] = th.tensor(data_label)
    hg.nodes['app'].data['train_mask'] = train_mask
    hg.nodes['app'].data['val_mask'] = val_mask
    hg.nodes['app'].data['test_mask'] = test_mask

    return hg, 'app', num_classes

def get_binary_mask(total_size, indices):
    mask = th.zeros(total_size)
    mask[indices] = 1
    return mask.to(th.bool)

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:
    :param shuffle:
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    return full_list[:offset], full_list[offset:]


def data_split_imb(full_list):
    list_a, list_b = full_list[1000:], full_list[:1000]
    random.shuffle(list_a)
    random.shuffle(list_b)
    list_tmp_a_train, list_tmp_a_other =  data_split(list_a, 0.7)
    list_tmp_b_train, list_tmp_b_other =  data_split(list_b, 0.7)
    return np.append(list_tmp_a_train, list_tmp_b_train), np.append(list_tmp_a_other, list_tmp_b_other)



# 将邻接矩阵转换为稀疏矩阵
def vert(adj_matrix):
    # adj_matrix 是邻接矩阵
    tmp_coo = scipy.sparse.coo_matrix(adj_matrix)
    row = th.from_numpy(tmp_coo.row)
    col = th.from_numpy(tmp_coo.col)
    return tuple((row, col))

# 将邻接矩阵转换为稀疏矩阵（反向）
def vertT(adj_matrix):
    # adj_matrix 是邻接矩阵
    tmp_coo = scipy.sparse.coo_matrix(adj_matrix)
    row = th.from_numpy(tmp_coo.row)
    col = th.from_numpy(tmp_coo.col)
    return tuple((col, row))