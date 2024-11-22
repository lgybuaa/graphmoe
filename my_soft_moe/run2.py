from models import GraphMOE
import sys
sys.path.append("..")
from TAGLAS import get_dataset
import os
from TAGLAS import get_task
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
from TAGLAS.tasks.text_encoder import SentenceEncoder
from params import args
from tqdm import tqdm


if len(args.gpu.split(',')) == 2:
    args.devices = ['cuda:0', 'cuda:1']
elif len(args.gpu.split(',')) > 2:
    raise Exception('Devices should be less than 2')
else:
    args.devices = [f'cuda:{args.gpu}']
# dict_keys(['book_children', 'book_history', 'computer', 'sports', 'photo', 'products', 'wikics', 'reddit', 'instagram'])
data_ecommercial = torch.load("/home/luguangyue/lgy/graph_MoE/dataset/graph_data_ecommercial.pt",weights_only=False)
# dict_keys(['arxiv', 'cora', 'pubmed', 'Industrial', 'cora_simple'])
data_paper = torch.load("/home/luguangyue/lgy/graph_MoE/dataset/graph_data_paper.pt",weights_only=False)
dataset_arxiv = data_paper['arxiv'].to(args.devices[0])
dataset_arxiv.y.squeeze_()
dataset_cora = data_paper['cora'].to(args.devices[0])
dataset_bookc = data_ecommercial['book_children'].to(args.devices[0])
for key in dataset_arxiv.keys():
    if key not in dataset_bookc.keys():
        dataset_arxiv[key] = None
    if key not in dataset_cora.keys():
        dataset_arxiv[key] = None
for key in dataset_cora.keys():
    if key not in dataset_arxiv.keys():
        dataset_cora[key] = None
    if key not in dataset_bookc.keys():
        dataset_cora[key] = None
arxiv_loader = LinkNeighborLoader(
    dataset_arxiv,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=256,
    edge_label_index=dataset_arxiv.edge_index,
    edge_label=torch.ones(dataset_arxiv.edge_index.size(1)),
    neg_sampling_ratio=0.5,
    shuffle=True,
)
cora_loader = LinkNeighborLoader(
    dataset_cora,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=256,
    edge_label_index=dataset_cora.edge_index,
    edge_label=torch.ones(dataset_cora.edge_index.size(1)),
    neg_sampling_ratio=0.5,
    shuffle=True,
)
bookc_loader = LinkNeighborLoader(
    dataset_bookc,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=256,
    edge_label_index=dataset_bookc.edge_index,
    edge_label=torch.ones(dataset_bookc.edge_index.size(1)),
    neg_sampling_ratio=0.5,
    shuffle=True,
)


# sampled_data = next(iter(loader)).to(args.devices[0])
# print(sampled_data)
# print("edge_label_index:",sampled_data.edge_label_index)
# print("input_idx:",sampled_data.input_id)
# print("edge_label:",sampled_data.edge_label)
# arxiv_node_task = get_task("arxiv", "subgraph_text", split="test", save_data=True, from_saved=True)
# arxiv_node_task.convert_text_to_embedding(encoder_name, encoder)
# batch = arxiv_node_task.collate([arxiv_node_task[i] for i in range(4)])
# to get node text features for all nodes in the batch
# x = batch.x[batch.node_map]
# to get edge text features for all edges in the batch
# edge_index = batch.edge_index

model = GraphMOE().to(args.devices[0])  # 实例化GCN模型，并移动到对应设备
# data = batch.to(args.devices[0])  # 获取数据集的第一个图数据，并移动到对应设备
# data.x = torch.randn((len(data.x),512)).to(args.devices[0])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # 定义Adam优化器
batch_len = min(len(arxiv_loader),len(cora_loader),len(bookc_loader))

for epoch in range(20):  # 进行训练循环，共200个epoch
    model.train()  # 将模型设置为训练模式
    loss_all = 0
    all_num = 0
    accuracy_all = 0
    for _ in tqdm(range(batch_len)):
        arxiv_batch_data = next(iter(arxiv_loader))
        cora_batch_data = next(iter(cora_loader))
        bookc_batch_data = next(iter(bookc_loader))
        # print("arxiv_batch_data:",arxiv_batch_data)
        # print("cora_batch_data:",cora_batch_data)
        # print("bookc_batch_data:",bookc_batch_data)
        list_train_data = [arxiv_batch_data,bookc_batch_data]
        batch_data = Batch.from_data_list(list_train_data)
        # print("batch_data:",batch_data)
        # print("edge_label:",batch_data.edge_label)
        # print("batch:",batch_data.batch)
        batch_data.to(args.devices[0])
        optimizer.zero_grad()  # 梯度清零
        out = model(batch_data)  # 前向传播，得到模型输出

        for i in range(len(list_train_data)):
            
        # print(out.size())
        # print(out)
        # logistic = F.sigmoid(out)  # 计算预测结果
        # print("logistic_size:",logistic.size())
        # print(logistic)
        # print("logistic device:",logistic.device)
        # print("edge_label device:",batch_data.edge_label.device)
        # loss = F.binary_cross_entropy(logistic, batch_data.edge_label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        predictions = torch.where(logistic > 0.5, 1, 0)
        score = torch.sum(torch.where(predictions == batch_data.edge_label, 1, 0))
        accuracy_all += score.item()
        loss_all += loss.item()*len(batch_data.edge_label)  # 累加损失值
        all_num += len(batch_data.edge_label)
    loss_mean = loss_all / all_num
    accuracy = accuracy_all / all_num
    print(f"Epoch {epoch}, Loss {loss_mean}")  # 输出当前epoch的损失值
    print("accuracy:",accuracy)
