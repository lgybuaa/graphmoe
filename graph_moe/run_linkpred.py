from models import GraphMOE, GCN_M
import sys
sys.path.append("..")
import os
import torch
import numpy as np
from params import args
from train_utils import train_epoch_link
from data_utils import get_link_datasets, get_link_dataloader
torch.set_printoptions(profile="full")

# data_ecommercial: dict_keys(['book_children', 'book_history', 'computer', 'sports', 'photo', 'products', 'wikics', 'reddit', 'instagram'])
# data_paper: dict_keys(['arxiv', 'cora', 'pubmed', 'Industrial', 'cora_simple'])

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
args.flagdir = 'log_link'

# device
args.devices = [f'cuda:{args.gpu}']

# 加载数据
train_datasets, test_datasets = get_link_datasets(args.train_datasets, args.test_datasets)
len_train_datasets = len(train_datasets)
len_test_datasets = len(test_datasets)
train_loader = get_link_dataloader(train_datasets, args.batch, args.workers, args.neg_sampling_ratio)


# 初始化模型和优化器
model = GCN_M().to(args.devices[0])  # 实例化GCN模型，并移动到对应设备
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 定义Adam优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1, betas=(0.9,0.95), eps=1e-8)  # 定义AdamW优化器
# 训练模型
metrics_dict = train_epoch_link(model, train_loader, test_datasets, optimizer, args.devices[0], args.epoch, len_train_datasets, len_test_datasets)

# best eval acc
best_eval_ndcg = max(metrics_dict['ndcg_eval'])
best_epoch  = metrics_dict['ndcg_eval'].index(best_eval_ndcg)


# 保存模型
torch.save(model.state_dict(), f'model/num_experts{args.expert_num}_topk{args.topk}_train_epoch{args.epoch}_batch{args.batch}_lr{args.lr}_lambda{args.lambda_inv}_inv{args.inv_fuc}_traindata {args.train_datasets}.pth')
with open("./model/record.txt", "a") as f:
    f.write(f"traindata {args.train_datasets}, testdata {args.test_datasets}, num_experts {args.expert_num}, topk {args.topk}, train_epoch {args.epoch}, batch {args.batch}, inv {args.inv_fuc}, lr {args.lr}, lambda {args.lambda_inv}, Loss {metrics_dict['loss_train'][-1]}, accuracy {metrics_dict['accuracy_train'][-1]},  ndcg_eval {metrics_dict['ndcg_eval'][-1]}, recall {metrics_dict['recall_eval'][-1]} best_eval_ndcg {best_eval_ndcg} best_epoch {best_epoch}\n")