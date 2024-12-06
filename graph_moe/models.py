import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from params import args
import numpy as np
from Utils.TimeLogger import log
from torch.nn import MultiheadAttention
from time import time
from torch_geometric.nn import GCNConv  # 从PyTorch几何库中导入图卷积网络层（GCNConv）
from torch_geometric.data import Data, Batch
from einops import rearrange
from torch import nn, einsum, Tensor
from einops import rearrange, pack, unpack
from my_gcn import My_GCNConv, My_GCNConv_2
from train_utils import LayerNorm, RMSNorm, gumbel_noise



class GCN(torch.nn.Module):  # 定义一个GNN类，继承自PyTorch的Module类
    def __init__(self):  # 定义GNN类的初始化函数
        super().__init__()  # 调用父类（Module类）的初始化函数
        # 创建第一个图卷积层，输入特征维度为数据集节点特征维度，输出特征维度为128
        self.conv1 = GCNConv(args.latdim, args.latdim*2)
        # 创建第二个图卷积层，输入特征维度为128，输出特征维度为数128
        self.conv2 = GCNConv(args.latdim*2, args.latdim*2)
        # 创建第二个图卷积层，输入特征维度为128，输出特征维度为数128
        self.conv3 = GCNConv(args.latdim*2, args.latdim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):  # 定义前向传播函数，接受一个数据对象作为输入
        x, edge_index = data[0], data[1]  # 从数据对象中获取节点特征和边索引
        x = self.conv1(x, edge_index)  # 通过第一个图卷积层处理节点特征
        x = F.relu(x)  # 对输出进行ReLU激活函数操作
        x = self.dropout(x)  # 对输出进行Dropout操作，用于防止过拟合
        # x = F.dropout(x, training=self.training)  # 对输出进行Dropout操作，用于防止过拟合
        x = self.conv2(x, edge_index)  # 通过第二个图卷积层处理节点特征
        x = F.relu(x)  # 对输出进行ReLU激活函数操作
        x = self.dropout(x)  # 对输出进行Dropout操作，用于防止过拟合
        # x = F.dropout(x, training=self.training)  # 对输出进行Dropout操作，用于防止过拟合
        x = self.conv3(x, edge_index)  # 通过第二个图卷积层处理节点特征
        x = F.relu(x)  # 对输出进行ReLU激活函数操作
        return x  # 对输出进行LogSoftmax操作，得到预测结果


class Experts(nn.Module):
    def __init__(
        self,
        experts,
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

    def forward(
        self,
        x,
        is_distributed = None
    ):
        """
        einops notation:
        b - batch
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """
        x, edge = x[0], x[1]    # (b, e, n, d), (b, 2, n)
        shape, num_experts = x.shape, self.num_experts  # b,e,n,d

        expert_slice = slice(0, num_experts)
        # x = rearrange(x, 'e n d -> e n d')
        # get the experts in use
        experts = self.experts[expert_slice]
        # route tokens to appropriate experts
        outs = []
        for expert, expert_input in zip(experts, x):
            out = expert([expert_input, edge])  # (b, n, d)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)    # (e, b, n, d)
        else:
            outs = torch.empty_like(x).requires_grad_()

        # all gather across merged expert batches dimensions
        # then split the batch dimension back
        # outs = rearrange(outs, 'e b n d -> b e n d')
        assert outs.shape == shape
        return outs


class GraphMOE(nn.Module):
    def __init__(self):
        super(GraphMOE, self).__init__()
        
        self.norm = LayerNorm(args.latdim)
        self.slot_norm = LayerNorm(args.latdim)

        self.softgate = nn.Parameter(t.randn(args.expert_num, args.latdim),requires_grad=True) # (e, d)
        self.experts = Experts(
            experts=[GCN() for _ in range(args.expert_num)],
        )
        self.lin_pred = nn.Linear(args.latdim, 1)

    def forward(self, batch_data: list):
        """
        einstein notation
        b - batch
        n - number of nodes
        e - number of experts
        d - feature dimension
        """
        # x = []
        # edge = []
        # for i in range(len(batch_data)):
        #     x.append(batch_data[i].x[batch_data[i].node_map])
        #     edge.append(batch_data[i].edge_index)
        # x = torch.stack(x)  # (b, n, d)
        # edge = torch.stack(edge)    # (b, 2, n)
        x = batch_data.x
        x = self.norm(x)
        # x = batch_data.x[batch_data.node_map]
        edge = batch_data.edge_index
        edge_label_index = batch_data.edge_label_index
        # x = rearrange(x, 'n d -> 1 n d')
        softgate = self.slot_norm(self.softgate)
        temperature = args.temperature
        logits = einsum('n d, e d -> n e', x, softgate)

        noise_mult = args.noise_mult
        if args.add_noise:
            noise = gumbel_noise(logits) * noise_mult
            logits = logits + noise
            
        logits = logits / temperature
        dispatch_weights = logits.softmax(dim=1)    # (b, n, e)
        # 结果聚合
        combine_weights = logits.softmax(dim=-1)    # (b, n, e)

        slots = einsum('n d, n e -> e n d', x, dispatch_weights)  # (b, e, n, d)

        expert_out = self.experts([slots,edge])    # (b, e, n, d)

        # 结果聚合
        out = einsum('e n d, n e -> n d', expert_out, combine_weights)

        # 链接预测
        if self.training:
            h_src = out[edge_label_index[0, :]]
            h_dst = out[edge_label_index[1, :]]
            src_dst_mult = h_src * h_dst
            out = self.lin_pred(src_dst_mult).squeeze(-1)
            return out
        else:
            h_src = out[edge_label_index[0, :]]
            h_dst = out[edge_label_index[1, :]]
            src_dst_mult = h_src * h_dst
            out = self.lin_pred(src_dst_mult).squeeze(-1)
            return out



        # return out

class GCN_M(torch.nn.Module):  # 定义一个GNN类，继承自PyTorch的Module类
    def __init__(self):  # 定义GNN类的初始化函数
        super().__init__()  # 调用父类（Module类）的初始化函数
        # 创建第一个图卷积层，输入特征维度为数据集节点特征维度，输出特征维度为128
        self.conv1 = My_GCNConv(args.latdim, args.latdim*2)
        # 创建第二个图卷积层，输入特征维度为128，输出特征维度为数128
        self.conv2 = My_GCNConv(args.latdim*2, args.latdim*2)
        # 创建第二个图卷积层，输入特征维度为128，输出特征维度为数128
        self.conv3 = My_GCNConv(args.latdim*2, args.latdim)

        self.lin_pred = nn.Linear(args.latdim, 1)

    def forward(self, data):  # 定义前向传播函数，接受一个数据对象作为输入
        x = data.x
        # x = batch_data.x[batch_data.node_map]
        edge_index = data.edge_index
        edge_label_index = data.edge_label_index

        x = self.conv1(x, edge_index)  # 通过第一个图卷积层处理节点特征
        # x = F.relu(x)  # 对输出进行ReLU激活函数操作
        x = F.dropout(x, training=self.training)  # 对输出进行Dropout操作，用于防止过拟合
        x = self.conv2(x, edge_index)  # 通过第二个图卷积层处理节点特征
        # x = F.relu(x)  # 对输出进行ReLU激活函数操作
        x = F.dropout(x, training=self.training)  # 对输出进行Dropout操作，用于防止过拟合
        out = self.conv3(x, edge_index)  # 通过第二个图卷积层处理节点特征
        # out = F.relu(x)  # 对输出进行ReLU激活函数操作

        h_src = out[edge_label_index[0, :]]
        h_dst = out[edge_label_index[1, :]]
        src_dst_mult = h_src * h_dst
        out = self.lin_pred(src_dst_mult).squeeze(-1)
        return out

class GCN_M_2(torch.nn.Module):  # 定义一个GNN类，继承自PyTorch的Module类
    def __init__(self):  # 定义GNN类的初始化函数
        super().__init__()  # 调用父类（Module类）的初始化函数
        # 创建第一个图卷积层，输入特征维度为数据集节点特征维度，输出特征维度为128
        self.conv1 = My_GCNConv_2(args.latdim, args.latdim*2)
        # 创建第二个图卷积层，输入特征维度为128，输出特征维度为数128
        self.conv2 = My_GCNConv_2(args.latdim*2, args.latdim*2)
        # 创建第三个图卷积层，输入特征维度为128，输出特征维度为数128
        self.conv3 = My_GCNConv_2(args.latdim*2, args.latdim)
        
        self.dropout = nn.Dropout(p=0.5)

        self.lin_pred = nn.Linear(args.latdim, 1)

    def forward(self, data):  # 定义前向传播函数，接受一个数据对象作为输入
        x = data.x
        # x = batch_data.x[batch_data.node_map]
        edge_index = data.edge_index
        edge_label_index = data.edge_label_index

        x = self.conv1(x, edge_index)  # 通过第一个图卷积层处理节点特征
        x = F.relu(x)  # 对输出进行ReLU激活函数操作
        x = self.dropout(x)
        # x = F.dropout(x, training=self.training)  # 对输出进行Dropout操作，用于防止过拟合
        x = self.conv2(x, edge_index)  # 通过第二个图卷积层处理节点特征
        x = F.relu(x)  # 对输出进行ReLU激活函数操作
        x = self.dropout(x)
        # x = F.dropout(x, training=self.training)  # 对输出进行Dropout操作，用于防止过拟合
        x = self.conv3(x, edge_index)  # 通过第二个图卷积层处理节点特征
        out = F.relu(x)  # 对输出进行ReLU激活函数操作

        h_src = out[edge_label_index[0, :]]
        h_dst = out[edge_label_index[1, :]]
        src_dst_mult = h_src * h_dst
        out = self.lin_pred(src_dst_mult).squeeze(-1)
        return out