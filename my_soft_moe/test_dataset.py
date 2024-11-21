import sys
sys.path.append("..")
from TAGLAS import get_dataset
import os
from TAGLAS import get_task
from torch_geometric.data import DataLoader
# 注意os.environ得在import huggingface库相关语句之前执行。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
dataset = get_dataset("cora_node")
data = dataset[0]

# print("图数据：", data)
# print("图数据节点长度：", len(data.x))
# print("样本的特征：", data.x[0])
# print("样本的边特征：", data.edge_attr)
# print("样本的node map 矩阵：", data.node_map)
# print("all labels:", data.label)
# print("edge index:", data.edge_index)   # shape:[2, num_edges]，第一行是源节点，第二行是目标节点
arxiv_node_task = get_task("arxiv", "subgraph_text", split="test", save_data=True, from_saved=True)
print(arxiv_node_task)

batch = arxiv_node_task.collate([arxiv_node_task[i] for i in range(3)])
# to get node text features for all nodes in the batch
x = batch.x[batch.node_map]
# to get edge text features for all edges in the batch
edge_attr = batch.edge_attr[batch.edge_map]
edge_index = batch.edge_index
# print(edge_index)