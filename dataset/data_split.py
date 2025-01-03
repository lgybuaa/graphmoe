import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RandomLinkSplit
import pickle
import scipy.sparse as sp
import numpy as np
import random
import os
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


def generate_random_id(sample_number):
    # 随机方式生成
    shuffled_idx = np.array(range(sample_number))
    np.random.shuffle(shuffled_idx)  # 已经被随机打乱
    train_idx = shuffled_idx[:int(0.8 * sample_number)].tolist()
    val_idx = shuffled_idx[int(0.8 * sample_number): int(0.9 * sample_number)].tolist()
    test_idx = shuffled_idx[int(0.9 * sample_number):].tolist()
    return train_idx, val_idx, test_idx


def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)

# 按节点划分数据集
def split_dataset_by_nodes(dataset):
    sample_num = len(dataset.y)
    train_id, val_id, test_id = generate_random_id(sample_num)
    train_mask, val_mask, test_mask = sample_mask(train_id, sample_num), sample_mask(val_id, sample_num), sample_mask(
        test_id, sample_num)
    dataset.train_mask = train_mask
    dataset.val_mask = val_mask
    dataset.test_mask = test_mask

# 按边划分数据集
def split_dataset_by_edges(dataset):
    transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False, disjoint_train_ratio=0,neg_sampling_ratio=0)
    train_data, val_data, test_data = transform(dataset)
    return train_data, val_data, test_data

# 存储边划分数据集
def save_dataset_by_edges(dataset, save_dir):
    train_data, val_data, test_data = split_dataset_by_edges(dataset)
    trn_mat, val_mat, tst_mat = train_data.edge_index, val_data.edge_label_index, test_data.edge_label_index
    trn_mat = sp.coo_matrix((np.ones(trn_mat.shape[1]), (trn_mat[0], trn_mat[1])),
                            shape=(dataset.num_nodes, dataset.num_nodes))
    val_mat = sp.coo_matrix((np.ones(val_mat.shape[1]), (val_mat[0], val_mat[1])),
                            shape=(dataset.num_nodes, dataset.num_nodes))
    tst_mat = sp.coo_matrix((np.ones(tst_mat.shape[1]), (tst_mat[0], tst_mat[1])),
                            shape=(dataset.num_nodes, dataset.num_nodes))
    feats = dataset.x.numpy()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + "/trn_mat.pkl", "wb") as f:
        pickle.dump(trn_mat, f)
    with open(save_dir + "/val_mat.pkl", "wb") as f:
        pickle.dump(val_mat, f)
    with open(save_dir + "/tst_mat.pkl", "wb") as f:
        pickle.dump(tst_mat, f)
    with open(save_dir + "/feats.pkl", "wb") as f:
        pickle.dump(feats, f)
    torch.save(train_data, save_dir + "/moe_train_data.pt")
    torch.save(val_data, save_dir + "/moe_val_data.pt")
    torch.save(test_data, save_dir + "/moe_test_data.pt")
    print(f"save {data_name} done")

def uniform_property(list_dataset):
    key_property = {"x", "edge_index", "y", "label_names"}
    for data_name, dataset in list_dataset.items():
        if len(dataset.y.shape) > 1:
            dataset.y = dataset.y.squeeze_()
    for data_name, dataset in list_dataset.items():
        for key in dataset.keys():
            if key not in key_property:
                dataset[key] = None
    return list_dataset


def load_one_file(filename):
    with open(filename, 'rb') as fs:
        ret = (pickle.load(fs) != 0).astype(np.float32)
    if type(ret) != coo_matrix:
        ret = sp.coo_matrix(ret)
    return ret


def load_feats(filename):
    try:
        with open(filename, 'rb') as fs:
            feats = pickle.load(fs)
    except Exception as e:
        print(filename + str(e))
        exit()
    return feats


# dict_keys(['book_children', 'book_history', 'computer', 'sports', 'photo', 'products', 'wikics', 'reddit', 'instagram'])
data_ecommercial = uniform_property(
    torch.load("/home/luguangyue/lgy/graph_MoE/dataset/graph_data_ecommercial.pt", weights_only=False))
# dict_keys(['arxiv', 'cora', 'pubmed', 'Industrial', 'cora_simple'])
data_paper = uniform_property(
    torch.load("/home/luguangyue/lgy/graph_MoE/dataset/graph_data_paper.pt", weights_only=False))

# for data_name, dataset in data_ecommercial.items():
#     print(data_name, dataset)
# for data_name, dataset in data_paper.items():
#     print(data_name, dataset)
# with open("./anygraph_datasets/arxiv/trn_mat.pkl", "rb") as f:
#     tst_mat = pickle.load(f)
#     print(tst_mat)
#     print(tst_mat.shape)
#     print(type(tst_mat))
# test_data = torch.load("./mygraph_datasets/arxiv/moe_test_data.pt", weights_only=False)
# print(test_data)


# print(data_paper["cora_simple"]["test_id"])
# for data_name, dataset in data_ecommercial.items():
# 按节点划分数据集
# for data_name, dataset in data_ecommercial.items():
#     split_dataset_by_nodes(dataset)
# for data_name, dataset in data_paper.items():
#     split_dataset_by_nodes(dataset)
# torch.save(data_ecommercial, "/home/luguangyue/lgy/graph_MoE/dataset/graph_data_ecommercial_node_split.pt")
# torch.save(data_paper, "/home/luguangyue/lgy/graph_MoE/dataset/graph_data_paper_node_split.pt")

# 按边划分数据集
for data_name, dataset in data_ecommercial.items():
    save_dir = "./mygraph_datasets/" + data_name
    save_dataset_by_edges(dataset, save_dir)
for data_name, dataset in data_paper.items():
    save_dir = "./mygraph_datasets/" + data_name
    save_dataset_by_edges(dataset, save_dir)

    

# for data_name, dataset in data_ecommercial.items():
#     print(data_name, dataset)
# for data_name, dataset in data_paper.items():
#     print(data_name, dataset)


