import dgl
import numpy as np
import random
import torch as th
import scipy
from scipy.sparse import csc_matrix



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

def load_drebin_mal(remove_self_loop):
    assert not remove_self_loop
    # 数据集路径
    dataset_path = './openhgnn/dataset/drebin/dataset.npy'

    dataset = np.load(dataset_path, allow_pickle=True).item()
    index = 0
    permssion_mtx = csc_matrix(dataset.get('permssion_mtx')[index:])
    intent_mtx = csc_matrix(dataset.get('intent_mtx')[index:])
    api_call_mtx = csc_matrix(dataset.get('api_call_mtx')[index:])
    activity_mtx = csc_matrix(dataset.get('activity_mtx')[index:])
    service_mtx = csc_matrix(dataset.get('service_mtx')[index:])
    permission_type_mtx = csc_matrix(dataset.get('permission_type_mtx')[index:])
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
        ('permission', 'ppt', 'permission_type'): vert(permission_type_mtx),
        ('permission_type', 'ptp', 'permission'): vertT(permission_type_mtx),
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



    num_classes = 2

    indices = np.arange(0, 12000 - index)
    random.shuffle(indices)

    train_idx, temp_idx = data_split(indices, 0.8)
    val_idx, test_idx = data_split(temp_idx, 0.5)
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