import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data

# dict_keys(['book_children', 'book_history', 'computer', 'sports', 'photo', 'products', 'wikics', 'reddit', 'instagram'])
data_ecommercial = torch.load("/home/luguangyue/lgy/graph_MoE/dataset/graph_data_ecommercial.pt", weights_only=False)
# dict_keys(['arxiv', 'cora', 'pubmed', 'Industrial', 'cora_simple'])
data_paper = torch.load("/home/luguangyue/lgy/graph_MoE/dataset/graph_data_paper.pt", weights_only=False)

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

def concat_graph(graph1,graph2):
    edge_index = torch.concat(
        [graph1.edge_index, graph2.edge_index + graph1.num_nodes],
        axis=1,
    )
    x = torch.concat([graph1.x, graph2.x])
    y = torch.concat([graph1.y, graph2.y])
    node_mask = torch.concat([graph1.node_mask, graph2.node_mask])
    data = Data(x=x, edge_index=edge_index, y=y, node_mask=node_mask)
    return data