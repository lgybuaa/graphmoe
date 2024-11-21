from models import GraphMOE
import sys
sys.path.append("..")
from TAGLAS import get_dataset
import os
from TAGLAS import get_task
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from TAGLAS.tasks.text_encoder import SentenceEncoder
from params import args
encoder_name = "llama3_8b"
encoder = SentenceEncoder(encoder_name)
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
# arxiv_node_task.convert_text_to_embedding(encoder_name, encoder)
batch = arxiv_node_task.collate([arxiv_node_task[i] for i in range(4)])
# to get node text features for all nodes in the batch
x = batch.x[batch.node_map]
# to get edge text features for all edges in the batch
edge_index = batch.edge_index

if len(args.gpu.split(',')) == 2:
    args.devices = ['cuda:0', 'cuda:1']
elif len(args.gpu.split(',')) > 2:
    raise Exception('Devices should be less than 2')
else:
    args.devices = [f'cuda:{args.gpu}']
model = GraphMOE().to(args.devices[0])  # 实例化GCN模型，并移动到对应设备
data = batch.to(args.devices[0])  # 获取数据集的第一个图数据，并移动到对应设备
data.x = torch.randn((len(data.x),512)).to(args.devices[0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # 定义Adam优化器
model.train()  # 将模型设置为训练模式
for epoch in range(1):  # 进行训练循环，共200个epoch
    optimizer.zero_grad()  # 梯度清零
    out = model(data)  # 前向传播，得到模型输出
    print(out.size())
    print(data.label_map[data.label_map.cpu().numpy()])
    print(data.target_index)
    print(data.batch)
    print(len(data[0].node_map))
# y是什么；label_map对应的是target_index吗