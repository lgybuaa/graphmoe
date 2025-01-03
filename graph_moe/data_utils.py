import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, LinkNeighborLoader
import numpy as np
import scipy.sparse as sp
from params import args

# dict_keys(['book_children', 'book_history', 'computer', 'sports', 'photo', 'products', 'wikics', 'reddit', 'instagram'])
data_ecommercial = torch.load("/home/luguangyue/lgy/graph_MoE/dataset/graph_data_ecommercial.pt", weights_only=False)
# dict_keys(['arxiv', 'cora', 'pubmed', 'Industrial', 'cora_simple'])
data_paper = torch.load("/home/luguangyue/lgy/graph_MoE/dataset/graph_data_paper.pt", weights_only=False)

data_all = {}
data_all.update(data_paper)
data_all.update(data_ecommercial)


def uniform_property(list_dataset):
    key_property = set(list_dataset[0].keys())
    for dataset in list_dataset:
        key_property = key_property & set(dataset.keys())
    for dataset in list_dataset:
        if len(dataset.y.shape) > 1:
            dataset.y = dataset.y.squeeze_()
    for dataset in list_dataset:
        for key in dataset.keys():
            if key not in key_property:
                dataset[key] = None
    return list_dataset


def concat_graph(graph1, graph2):
    edge_index = torch.concat(
        [graph1.edge_index, graph2.edge_index + graph1.num_nodes],
        axis=1,
    )
    x = torch.concat([graph1.x, graph2.x])
    y = torch.concat([graph1.y, graph2.y])
    node_mask = torch.concat([graph1.node_mask, graph2.node_mask])
    if 'edge_label_index' in graph1.keys() and 'edge_label_index' in graph2.keys():
        edge_label_index = torch.concat(
            [graph1.edge_label_index, graph2.edge_label_index + graph1.num_nodes],
            axis=1,
        )
        edge_label = torch.concat([graph1.edge_label, graph2.edge_label])
        data = Data(x=x, edge_index=edge_index, y=y, node_mask=node_mask, edge_label_index=edge_label_index, edge_label=edge_label)
    else:
        data = Data(x=x, edge_index=edge_index, y=y, node_mask=node_mask)
    return data


def get_datasets(trn_datasets_names, tst_datasets_names):
    train_datasets_key = trn_datasets_names.split(',')
    test_datasets_key = tst_datasets_names.split(',')

    train_datasets = [data_all[key] for key in train_datasets_key]
    test_datasets = [data_all[key] for key in test_datasets_key]

    train_datasets = uniform_property(train_datasets)
    test_datasets = uniform_property(test_datasets)

    return train_datasets, test_datasets


def get_dataloader(datasets, num_batch, num_workers, neg_sampling_ratio):
    for i, data in enumerate(datasets):
        data.node_mask = torch.tensor([i] * data.y.shape[0])
    dataset = datasets[0]
    for data in datasets[1:]:
        dataset = concat_graph(dataset, data)

    data_loader = LinkNeighborLoader(
        dataset,
        # Sample 30 neighbors for each node for 3 iterations
        num_neighbors=[30] * 3,
        # Use a batch size of 128 for sampling training nodes
        batch_size=num_batch,
        edge_label_index=dataset.edge_index,
        edge_label=torch.ones(dataset.edge_index.size(1)),
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    return data_loader

def load_dataset(dataset_name):
    file_dir = args.data_dir + f'{dataset_name}/'
    trn_file = file_dir + 'moe_train_data.pt'
    tst_file = file_dir + 'moe_test_data.pt'
    val_file = file_dir + 'moe_val_data.pt'
    trn_data = torch.load(trn_file, weights_only=False)
    tst_data = torch.load(tst_file, weights_only=False)
    val_data = torch.load(val_file, weights_only=False)
    return trn_data, tst_data, val_data


def get_link_datasets(trn_datasets_names, tst_datasets_names):
    train_datasets_key = trn_datasets_names.split(',')
    test_datasets_key = tst_datasets_names.split(',')

    train_datasets = []
    for trn_name in train_datasets_key:
        trn_data, _, _ = load_dataset(trn_name)
        train_datasets.append(trn_data)

    test_datasets = {}
    for tst_name in test_datasets_key:
        trn_data, tst_data, val_data = load_dataset(tst_name)
        test_datasets[tst_name] = {"train": trn_data, "test": tst_data, "val": val_data}

    train_datasets = uniform_property(train_datasets)

    return train_datasets, test_datasets

def get_link_dataloader(datasets, num_batch, num_workers, neg_sampling_ratio):
    for i, data in enumerate(datasets):
        data.node_mask = torch.tensor([i] * data.y.shape[0])

    dataset = datasets[0]
    for data in datasets[1:]:
        dataset = concat_graph(dataset, data)

    data_loader = LinkNeighborLoader(
        dataset,
        # Sample 30 neighbors for each node for 3 iterations
        num_neighbors=[30] * 3,
        # Use a batch size of 128 for sampling training nodes
        batch_size=num_batch,
        neg_sampling_ratio=neg_sampling_ratio,
        edge_label_index=dataset.edge_label_index,
        edge_label=dataset.edge_label,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    return data_loader


class TstData(torch.utils.data.Dataset):
    def __init__(self, tst_data, trn_data):
        self.csrmat = sp.csr_matrix((np.ones(trn_data.edge_index.shape[1]), (trn_data.edge_index[0], trn_data.edge_index[1])),
                                shape=(trn_data.num_nodes, trn_data.num_nodes))
        tstLocs = [None] * tst_data.num_nodes
        tst_nodes = set()
        for i in range(tst_data.edge_label_index.shape[1]):
            row = tst_data.edge_label_index[0][i]
            col = tst_data.edge_label_index[1][i]
            if tst_data.edge_label[i] == 1:
                if tstLocs[row] is None:
                    tstLocs[row] = list()
                    tstLocs[row].append(col)
                tst_nodes.add(row)
        tst_nodes = np.array(list(tst_nodes))
        self.tst_nodes = tst_nodes
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tst_nodes)

    def __getitem__(self, idx):
        return self.tst_nodes[idx]
