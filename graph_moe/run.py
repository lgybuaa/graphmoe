from models import GraphMOE, GCN_M, GCN_M_2
import sys

sys.path.append("..")
from TAGLAS import get_dataset
import os
from TAGLAS import get_task
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd
from TAGLAS.tasks.text_encoder import SentenceEncoder
from params import args
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from train_utils import mean_nll, penalty, penalty_rex
from data_utils import data_ecommercial, data_paper, uniform_property, concat_graph
torch.set_printoptions(profile="full")
# data_ecommercial: dict_keys(['book_children', 'book_history', 'computer', 'sports', 'photo', 'products', 'wikics', 'reddit', 'instagram'])
#'photo', 'products'， 'book_children', 'book_history'
# 'arxiv', 'cora', 'pubmed'
# link1 :products,book_children,book_history,pubmed
# link2 :photo,arxiv,cora
# data_paper: dict_keys(['arxiv', 'cora', 'pubmed', 'Industrial', 'cora_simple'])
# datasets['link1'] = [
#         'products_tech', 'yelp2018', 'yelp_textfeat', 'products_home', 'steam_textfeat', 'amazon_textfeat', 'amazon-book', 'citation-2019', 'citation-classic', 'pubmed', 'citeseer', 'ppa', 'p2p-Gnutella06', 'soc-Epinions1', 'email-Enron',
#         ]
# datasets['link2'] = [
#     'Photo', 'Goodreads', 'Fitness', 'ml1m', 'ml10m', 'gowalla', 'arxiv', 'arxiv-ta', 'cora', 'CS', 'collab', 'proteins_spec0', 'proteins_spec1', 'proteins_spec2', 'proteins_spec3', 'ddi', 'web-Stanford', 'roadNet-PA',
#     ]

# 设置随机种子以保证结果可复现
# torch.manual_seed(42)
# np.random.seed(42)

if len(args.gpu.split(',')) == 2:
    args.devices = ['cuda:0', 'cuda:1']
elif len(args.gpu.split(',')) > 2:
    raise Exception('Devices should be less than 2')
else:
    args.devices = [f'cuda:{args.gpu}']

data_all = {}
data_all.update(data_paper)
data_all.update(data_ecommercial)

train_datasets_key = args.train_datasets.split(',')
test_datasets_key = args.test_datasets.split(',')

train_datasets = [data_all[key] for key in train_datasets_key]
test_datasets = [data_all[key] for key in test_datasets_key]

train_datasets = uniform_property(train_datasets)
test_datasets = uniform_property(test_datasets)
print(test_datasets)

for i, train_data in enumerate(train_datasets):
    train_data.node_mask = torch.tensor([i] * train_data.y.shape[0])
for i, test_data in enumerate(test_datasets):
    test_data.node_mask = torch.tensor([i] * test_data.y.shape[0])

train_dataset = train_datasets[0]
test_dataset = test_datasets[0]
for data in train_datasets[1:]:
    train_dataset = concat_graph(train_dataset, data)
for data in test_datasets[1:]:
    test_dataset = concat_graph(test_dataset, data)

train_loader = LinkNeighborLoader(
    train_dataset,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=args.batch,
    edge_label_index=train_dataset.edge_index,
    edge_label=torch.ones(train_dataset.edge_index.size(1)),
    neg_sampling_ratio=1,
    shuffle=True,
    num_workers=args.workers,
    persistent_workers=True,
)
test_loader = LinkNeighborLoader(
    test_dataset,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=args.batch,
    edge_label_index=test_dataset.edge_index,
    edge_label=torch.ones(test_dataset.edge_index.size(1)),
    neg_sampling_ratio=1,
    shuffle=True,
    num_workers=args.workers,
    persistent_workers=True,
)


model = GCN_M().to(args.devices[0])  # 实例化GCN模型，并移动到对应设备

# model = GCN_M_2().to(args.devices[0])

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)  # 定义Adam优化器

for epoch in range(args.epoch):  # 进行训练循环，共200个epoch
    model.train()  # 将模型设置为训练模式
    loss_mean = 0
    all_num = 0
    list_num = [0]*len(train_datasets)
    list_accuracy = [0] * len(train_datasets)
    batch_num = 0
    for train_data in tqdm(train_loader):  # 遍历每个batch
        # 标记边属于哪个数据集
        train_data.edge_label_mask = torch.tensor([0]*train_data.edge_label_index.shape[1])
        for i in range(len(train_data.edge_label_index[0])):
            src_node = train_data.edge_label_index[0][i]
            edge_mask_s = train_data.node_mask[src_node]
            train_data.edge_label_mask[i] = edge_mask_s

        train_data = train_data.to(args.devices[0])  # 将数据移动到对应设备
        # 前向传播，得到模型输出
        out = model(train_data)
        if batch_num % 1000 == 0:
            print(f"batch {batch_num}")
            # print(model.conv1.dispatch_weights)
            print(model.conv1.combine_weights[:, :100])
        batch_num += 1

        logistic = F.sigmoid(out)
        predictions = torch.where(logistic > 0.5, 1, 0)
        # 存储每个dataset的损失值
        list_loss = []
        # 存储每个dataset的惩罚项
        list_penalty = []
        for i in range(len(train_datasets)):
            edge_label_one = train_data.edge_label[train_data.edge_label_mask == i]
            out_one = out[train_data.edge_label_mask == i]
            logistic_one = logistic[train_data.edge_label_mask == i]
            predictions_one = predictions[train_data.edge_label_mask == i]
            # 计算损失值
            loss_one = F.binary_cross_entropy(logistic_one, edge_label_one)
            list_loss.append(loss_one)
            # 计算惩罚项
            if args.inv_fuc == "irm":
                # IRM
                list_penalty.append(penalty(logistic_one, edge_label_one))
            elif args.inv_fuc == 'rex':
                # REx
                list_penalty.append(penalty_rex(logistic_one, edge_label_one))

            # 计算准确率
            score = torch.sum(
                torch.where(predictions_one == edge_label_one, 1, 0))
            list_accuracy[i] += score.item()
            list_num[i] += torch.sum(train_data.edge_label_mask == i)

        if args.inv_fuc == "irm":
            # IRM
            train_penalty = torch.stack(list_penalty).mean()
        elif args.inv_fuc == 'rex':
            # REx
            train_penalty = torch.stack(list_penalty).var()

        train_loss = F.binary_cross_entropy(logistic, train_data.edge_label)
        lambda_inv = 1 if epoch <= 1 else args.lambda_inv
        loss_end = train_loss + lambda_inv * train_penalty
        if lambda_inv > 1.0:
            loss_end /= lambda_inv



        optimizer.zero_grad()  # 梯度清零
        loss_end.backward()  # 反向传播
        optimizer.step()  # 更新参数


        loss_mean += loss_end.item() * len(train_data.edge_label)  # 累加损失值
        all_num += len(train_data.edge_label)
    loss_mean = loss_mean / all_num
    assert all_num == sum(list_num)
    accuracy = sum(list_accuracy) / all_num
    accuracy_all = [list_accuracy[i] / list_num[i] for i in range(len(train_datasets))]
    print(
        f"Epoch {epoch}, Loss {loss_mean}, accuracy {accuracy}, accuracy_all {accuracy_all}")  # 输出当前epoch的损失值与准确率

    model.eval()
    with torch.no_grad():
        all_num_eval = 0
        list_num_eval = [0] * len(test_datasets)
        accuracy_eval = 0
        list_accuracy_eval = [0] * len(test_datasets)
        loss_eval = 0
        list_loss_eval = []
        list_y_true = [[] for _ in range(len(test_datasets))]
        list_y_pred = [[] for _ in range(len(test_datasets))]
        list_logistic_pred = [[] for _ in range(len(test_datasets))]
        y_true = []
        y_pred = []
        logistic_pred = []
        test_batch_num = 0
        for test_data in tqdm(test_loader):
            # 标记边属于哪个数据集
            test_data.edge_label_mask = torch.tensor([0] * test_data.edge_label_index.shape[1])
            for i in range(len(test_data.edge_label_index[0])):
                src_node = test_data.edge_label_index[0][i]
                edge_mask_s = test_data.node_mask[src_node]
                test_data.edge_label_mask[i] = edge_mask_s

            test_data = test_data.to(args.devices[0])
            # 前向传播，得到模型输出
            out = model(test_data)

            if test_batch_num % 500 == 0:
                print(f"test_batch {test_batch_num}")
                print(model.conv1.combine_weights)
            test_batch_num += 1

            logistic = F.sigmoid(out)
            predictions = torch.where(logistic > 0.5, 1, 0)

            y_true.extend(test_data.edge_label.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            logistic_pred.extend(logistic.cpu().numpy())
            for i in range(len(test_datasets)):
                edge_label_one = test_data.edge_label[test_data.edge_label_mask == i]
                out_one = out[test_data.edge_label_mask == i]
                logistic_one = logistic[test_data.edge_label_mask == i]
                predictions_one = predictions[test_data.edge_label_mask == i]
                # 计算损失值
                loss_one = F.binary_cross_entropy(logistic_one, edge_label_one)
                list_loss_eval.append(loss_one)
                # 计算准确率
                score = torch.sum(
                    torch.where(predictions_one == edge_label_one, 1, 0))
                list_accuracy_eval[i] += score.item()
                list_num_eval[i] += torch.sum(test_data.edge_label_mask == i)

                list_y_true[i].extend(edge_label_one.cpu().numpy())
                list_y_pred[i].extend(predictions_one.cpu().numpy())
                list_logistic_pred[i].extend(logistic_one.cpu().numpy())

            loss_eval += F.binary_cross_entropy(logistic,test_data.edge_label).item() * len(
                test_data.edge_label)
            score = torch.sum(torch.where(predictions == test_data.edge_label, 1, 0))
            accuracy_eval += score.item()
            all_num_eval += len(test_data.edge_label)

        accuracy_eval = accuracy_eval / all_num_eval
        loss_eval = loss_eval / all_num_eval
        list_accuracy_eval = [list_accuracy_eval[i] / list_num_eval[i] for i in range(len(test_datasets))]
        f1score_eval = f1_score(y_true, y_pred)
        auc_eval = roc_auc_score(y_true, logistic_pred)
        list_f1score_eval = [f1_score(list_y_true[i], list_y_pred[i]) for i in range(len(test_datasets))]
        list_auc_eval = [roc_auc_score(list_y_true[i], list_logistic_pred[i]) for i in range(len(test_datasets))]

        print(f"Eval, Loss {loss_eval}, accuracy {accuracy_eval} list_accuracy {list_accuracy_eval}\nf1score {f1score_eval} auc {auc_eval} list_f1score {list_f1score_eval} list_auc {list_auc_eval}")


torch.save(model.state_dict(), f'model/train_epoch{args.epoch}_batch{args.batch}_lr{args.lr}_lambda{args.lambda_inv}_inv{args.inv_fuc}_traindata {args.train_datasets}.pth')
with open("./model/record.txt", "a") as f:
    f.write(f"traindata {args.train_datasets}, testdata {args.test_datasets}, num_experts {args.expert_num}, train_epoch {args.epoch}, batch {args.batch}, inv {args.inv_fuc}, lr {args.lr}, lambda {args.lambda_inv}, Loss {loss_mean}, accuracy {accuracy}, loss_eval {loss_eval}, accuracy_eval {accuracy_eval}, f1_score {f1score_eval}\n")