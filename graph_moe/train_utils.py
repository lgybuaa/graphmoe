import torch
import torch.nn.functional as F
from torch import autograd
from torch import nn, einsum, Tensor
from torch.nn import Module
from typing import Optional, Tuple, List, Union
from params import args
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np


def mean_nll(logits, y):
    return F.binary_cross_entropy(logits, y)


def penalty(logits, y):
    scale = torch.tensor(1.).requires_grad_().to(args.devices[0])
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def penalty_rex(logits, y):
    scale = torch.tensor(1.).requires_grad_().to(args.devices[0])
    loss = mean_nll(logits * scale, y)
    return loss


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma


def l2norm(t):
    return F.normalize(t, dim=- 1)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def add_aux_loss(loss, all_router_logits):
    router_logits = ()
    for router_logits_one in all_router_logits:
        router_logits += (router_logits_one,)
    temporal_aux_loss = load_balancing_loss_func(
        router_logits,
        top_k=args.topk,
        num_experts=args.expert_num,
    )
    train_loss = loss + args.router_aux_loss_factor * temporal_aux_loss.to(loss.device)
    return train_loss


def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        top_k: int,
        num_experts: int = None,
) -> torch.Tensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor], List[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        top_k (`int`)
            Selected Top k over the experts.
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
        return 0.0

    compute_device = gate_logits[0].device
    # concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    #
    # routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    #
    # _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    #
    # expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    #
    # # Compute the percentage of tokens routed to each expert
    # tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    #
    # # Compute the average probability of routing to these experts
    # router_prob_per_expert = torch.mean(routing_weights, dim=0)
    #
    # overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

    list_topk = [torch.topk(F.softmax(gate_logits[i].to(compute_device), dim=1), top_k, dim=-1) for i in
                 range(len(gate_logits))]
    list_selected_experts = list(zip(*list_topk))[1]
    list_expert_mask = [torch.nn.functional.one_hot(selected_experts, num_classes=args.expert_num) for selected_experts
                        in list_selected_experts]
    # Compute the percentage of tokens routed to each expert
    list_tokens_per_expert = [torch.mean(expert_mask.float(), dim=0) for expert_mask in list_expert_mask]
    # Compute the average probability of routing to these experts
    list_router_prob_per_expert = [torch.mean(F.softmax(gate_logits[i], dim=1), dim=0) for i in
                                   range(len(gate_logits))]
    list_overall_loss = [torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0)) for
                         tokens_per_expert, router_prob_per_expert in
                         zip(list_tokens_per_expert, list_router_prob_per_expert)]
    overall_loss = sum(list_overall_loss)/len(list_overall_loss)

    return overall_loss * num_experts


def cal_datasets_loss(train_data, len_train_datasets, out, logistic, predictions):
    # 存储每个dataset的损失值
    list_loss = []
    # 存储每个dataset的惩罚项
    list_penalty = []
    list_num = [0] * len_train_datasets  # 记录每个数据集的样本数
    list_accuracy = [0] * len_train_datasets  # 记录每个数据集的准确率

    for i in range(len_train_datasets):
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

    return list_loss, list_accuracy, list_num, train_penalty


def print_log(batch_num, all_router_logits, is_train=True):

    list_topk = [torch.topk(F.softmax(all_router_logits[i], dim=1), args.topk, dim=-1) for i in range(len(all_router_logits))]
    list_selected_experts = list(zip(*list_topk))[1]
    list_expert_mask = [torch.nn.functional.one_hot(selected_experts, num_classes=args.expert_num) for selected_experts in list_selected_experts]
    # Compute the percentage of tokens routed to each expert
    list_tokens_per_expert = [torch.mean(expert_mask.float(), dim=0) for expert_mask in list_expert_mask]
    # Compute the average probability of routing to these experts
    list_router_prob_per_expert = [torch.mean(F.softmax(all_router_logits[i], dim=1), dim=0) for i in range(len(all_router_logits))]
    list_overall_loss = [torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0)) for tokens_per_expert, router_prob_per_expert in zip(list_tokens_per_expert, list_router_prob_per_expert)]
    list_expert_mask_sum = [expert_mask.sum(dim=0) for expert_mask in list_expert_mask]
    with open(
            f"{args.flagdir}/num_experts{args.expert_num}_topk{args.topk}_train_epoch{args.epoch}_batch{args.batch}_lr{args.lr}_lambda{args.lambda_inv}_inv{args.inv_fuc}_balance{args.router_aux_loss_factor}.txt",
            "a") as f:
        if is_train:
            f.write(f"batch {batch_num}\n")
        else:
            f.write(f"test batch {batch_num}\n")
        for i in range(len(list_expert_mask_sum)):
            f.write(f"expert_sum_{i}:\n\t{list_expert_mask_sum[i]}\n")
        for i in range(len(list_tokens_per_expert)):
            print(f"tokens_per_expert_{i}:\n\t", list_tokens_per_expert[i].tolist())
            print(f"router_prob_per_expert_{i}:\n\t", list_router_prob_per_expert[i].tolist())
            print(f"overall_loss_{i}:\n\t", list_overall_loss[i].item())
            print(f"loss_balance_{i}:\n\t", args.router_aux_loss_factor * list_overall_loss[i].item() * args.expert_num)
            print("\n")
        # print("the grad of expert2:", model.conv1.lin_moe.experts[2].lin.weight.grad)
        # f.write(f"{F.softmax(all_router_logits[0], dim=1)}\n")


def make_trn_masks(numpy_usrs, csr_mat):
    trn_masks = csr_mat[numpy_usrs].tocoo()
    cand_size = trn_masks.shape[1]
    trn_masks = t.from_numpy(np.stack([trn_masks.row, trn_masks.col], axis=0)).long()
    return trn_masks, cand_size

def calc_recall_ndcg(topLocs, tstLocs, batIds):
    assert topLocs.shape[0] == len(batIds)
    allRecall = allNdcg = 0
    for i in range(len(batIds)):
        temTopLocs = list(topLocs[i])
        temTstLocs = tstLocs[batIds[i]]
        tstNum = len(temTstLocs)
        maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk_pred))])
        recall = dcg = 0
        for val in temTstLocs:
            if val in temTopLocs:
                recall += 1
                dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
        recall = recall / tstNum
        ndcg = dcg / maxDcg
        allRecall += recall
        allNdcg += ndcg
    return allRecall, allNdcg

def train_batch(model, train_loader, optimizer, device, epoch_id, len_train_datasets):
    model.train()  # 将模型设置为训练模式
    loss_mean = 0  # 记录每个epoch的平均损失值
    loss_balance = 0  # 记录每个epoch的平衡损失值
    all_num = 0  # 记录每个epoch的总样本数
    batch_num = 0  # 记录batch数
    list_num = [0] * len_train_datasets  # 记录每个数据集的样本数
    list_accuracy = [0] * len_train_datasets  # 记录每个数据集的准确率

    for train_data in tqdm(train_loader):  # 遍历每个batch
        # 标记edge属于哪个数据集
        train_data.edge_label_mask = torch.tensor([0] * train_data.edge_label_index.shape[1])
        for i in range(len(train_data.edge_label_index[0])):
            src_node = train_data.edge_label_index[0][i]
            edge_mask_s = train_data.node_mask[src_node]
            train_data.edge_label_mask[i] = edge_mask_s

        train_data = train_data.to(device)  # 将数据移动到对应设备
        # 前向传播，得到模型输出
        out, all_router_logits, _ = model(train_data)

        # 输出batch的路由概率
        if args.flag_router:
            if batch_num % 1000 == 0:
                print_log(batch_num, all_router_logits)
        batch_num += 1

        # 计算任务损失
        logistic = F.sigmoid(out)
        train_loss_task = F.binary_cross_entropy(logistic, train_data.edge_label)
        # 添加路由平衡损失
        train_loss = add_aux_loss(train_loss_task, all_router_logits)
        loss_balance_one = train_loss - train_loss_task

        predictions = torch.where(logistic > 0.5, 1, 0)
        # 计算每个数据集的损失值、准确率、样本数、惩罚项
        list_loss, list_accuracy_batch, list_num_batch, train_penalty = cal_datasets_loss(train_data,
                                                                                          len_train_datasets, out,
                                                                                          logistic,
                                                                                          predictions)
        # 计算总损失
        lambda_inv = 1 if epoch_id <= 1 else args.lambda_inv
        loss_end = train_loss + lambda_inv * train_penalty
        if lambda_inv > 1.0:
            loss_end /= lambda_inv

        optimizer.zero_grad()  # 梯度清零
        loss_end.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 累加每个数据集的样本数、准确率
        for i in range(len_train_datasets):
            list_num[i] += list_num_batch[i]
            list_accuracy[i] += list_accuracy_batch[i]
        # 累加任务损失值
        loss_mean += train_loss_task.item() * len(train_data.edge_label)
        loss_balance += loss_balance_one.item() * len(train_data.edge_label)
        all_num += len(train_data.edge_label)
    loss_mean = loss_mean / all_num
    loss_balance = loss_balance / all_num
    assert all_num == sum(list_num)
    accuracy = sum(list_accuracy) / all_num
    accuracy_datasets = [list_accuracy[i] / list_num[i] for i in range(len_train_datasets)]
    print(
        f"Epoch {epoch_id}, Loss {loss_mean}, loss_balance {loss_balance} accuracy {accuracy}, accuracy_all {accuracy_datasets}")  # 输出当前epoch的损失值与准确率
    return loss_mean, accuracy


def eval(model, test_loader, device, len_test_datasets):
    model.eval()
    with torch.no_grad():
        all_num_eval = 0  # 记录总样本数
        list_num_eval = [0] * len_test_datasets  # 记录每个数据集的样本数

        accuracy_eval = 0  # 记录总准确率
        list_accuracy_eval = [0] * len_test_datasets  # 记录每个数据集的准确率

        loss_eval = 0  # 记录总损失值

        list_y_true = [[] for _ in range(len_test_datasets)]  # 记录每个数据集的真实标签
        list_y_pred = [[] for _ in range(len_test_datasets)]  # 记录每个数据集的预测标签
        list_logistic_pred = [[] for _ in range(len_test_datasets)]  # 记录每个数据集的预测概率
        y_true = []  # 记录总的真实标签
        y_pred = []  # 记录总的预测标签
        logistic_pred = []  # 记录总的预测概率
        test_batch_num = 0  # 记录batch数
        for test_data in tqdm(test_loader):
            # 标记边属于哪个数据集
            test_data.edge_label_mask = torch.tensor([0] * test_data.edge_label_index.shape[1])
            for i in range(len(test_data.edge_label_index[0])):
                src_node = test_data.edge_label_index[0][i]
                edge_mask_s = test_data.node_mask[src_node]
                test_data.edge_label_mask[i] = edge_mask_s

            test_data = test_data.to(device)
            # 前向传播，得到模型输出
            out, all_router_logits,_ = model(test_data)

            # 输出batch的路由概率
            if args.flag_router:
                if test_batch_num % 500 == 0:
                    print_log(test_batch_num, all_router_logits, is_train=False)
                    # _, selected_experts = torch.topk(F.softmax(all_router_logits[0], dim=1), args.topk, dim=-1)
                    # expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=args.expert_num)
                    # expert_mask_sum = expert_mask.sum(dim=0)
                    # expert_mask = expert_mask.sum(dim=1)
                    # with open(
                    #         f"log/num_experts{args.expert_num}_topk{args.topk}_train_epoch{args.epoch}_batch{args.batch}_lr{args.lr}_lambda{args.lambda_inv}_inv{args.inv_fuc}_balance{args.router_aux_loss_factor}.txt",
                    #         "a") as f:
                    #     f.write(f"test_batch {test_batch_num}\n")
                    #     f.write(f"expert_sum: {expert_mask_sum}\n")
                    #     f.write(f"{expert_mask}\n")
            test_batch_num += 1

            logistic = F.sigmoid(out)
            predictions = torch.where(logistic > 0.5, 1, 0)

            # 累加总的真实标签、预测标签、预测概率
            y_true.extend(test_data.edge_label.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            logistic_pred.extend(logistic.cpu().numpy())

            loss_eval += F.binary_cross_entropy(logistic, test_data.edge_label).item() * len(
                test_data.edge_label)
            score = torch.sum(torch.where(predictions == test_data.edge_label, 1, 0))
            accuracy_eval += score.item()
            all_num_eval += len(test_data.edge_label)
            # 计算数据集的指标
            for i in range(len_test_datasets):
                edge_label_one = test_data.edge_label[test_data.edge_label_mask == i]
                out_one = out[test_data.edge_label_mask == i]
                logistic_one = logistic[test_data.edge_label_mask == i]
                predictions_one = predictions[test_data.edge_label_mask == i]
                # 计算数据集准确率
                score = torch.sum(
                    torch.where(predictions_one == edge_label_one, 1, 0))
                list_accuracy_eval[i] += score.item()
                list_num_eval[i] += torch.sum(test_data.edge_label_mask == i)
                list_y_true[i].extend(edge_label_one.cpu().numpy())
                list_y_pred[i].extend(predictions_one.cpu().numpy())
                list_logistic_pred[i].extend(logistic_one.cpu().numpy())

        # 计算f1score、auc
        f1score_eval = f1_score(y_true, y_pred)
        auc_eval = roc_auc_score(y_true, logistic_pred)
        accuracy_eval = accuracy_eval / all_num_eval
        loss_eval = loss_eval / all_num_eval
        list_accuracy_eval = [list_accuracy_eval[i] / list_num_eval[i] for i in range(len_test_datasets)]
        list_f1score_eval = [f1_score(list_y_true[i], list_y_pred[i]) for i in range(len_test_datasets)]
        list_auc_eval = [roc_auc_score(list_y_true[i], list_logistic_pred[i]) for i in range(len_test_datasets)]

        print(
            f"Eval, Loss {loss_eval}, accuracy {accuracy_eval} list_accuracy {list_accuracy_eval}\nf1score {f1score_eval} auc {auc_eval} list_f1score {list_f1score_eval} list_auc {list_auc_eval}")
    return loss_eval, accuracy_eval, f1score_eval

def eval_link(model, test_dataset, dataset_name):
    trn_data = test_dataset["train"]
    tst_data = test_dataset["test"]
    trn_data_loader = LinkNeighborLoader(
        trn_data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 3,
        # Use a batch size of 128 for sampling training nodes
        batch_size=args.batch,
        neg_sampling_ratio=0,
        num_workers=workers,
        shuffle=False,
        persistent_workers=True,
    )

    feats = []
    test_batch_num = 0
    for train_data in tqdm(trn_data_loader):  # 遍历每个batch
        train_data = train_data.to(args.devices[0])  # 将数据移动到对应设备
        # 前向传播，得到模型输出
        _, all_router_logits, embeds = model(train_data)
        feats.append(embeds)
        if args.flag_router:
            if test_batch_num % 500 == 0:
                print(dataset_name)
                print_log(0, all_router_logits, is_train=False)
        test_batch_num += 1
    feats = torch.cat(feats, dim=0)

    test_data = Tst_Dataset(tst_data, trn_data)
    tst_loader = torch.utils.data.DataLoader(test_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        ep_recall, ep_ndcg = 0, 0
        ep_tstnum = len(tst_loader.dataset)
        steps = max(ep_tstnum // args.tst_batch, 1)
        for i, batch_data in enumerate(tst_loader):
            if args.tst_steps != -1 and i > args.tst_steps:
                break

            usrs = batch_data.long()
            trn_masks, cand_size = make_trn_masks(batch_data.numpy(), tst_loader.dataset.csrmat)
            all_preds = model.pred_for_test((usrs, trn_masks), cand_size, feats,)
            _, top_locs = torch.topk(all_preds, args.topk_pred)
            top_locs = top_locs.cpu().numpy()
            recall, ndcg = calc_recall_ndcg(top_locs, tst_loader.dataset.tstLocs, usrs)
            ep_recall += recall
            ep_ndcg += ndcg
    ret = dict()
    ret['Recall'] = ep_recall / ep_tstnum
    ret['NDCG'] = ep_ndcg / ep_tstnum
    ret['tstNum'] = ep_tstnum
    torch.cuda.empty_cache()
    return ret


def train_epoch(model, train_loader, test_loader, optimizer, device, num_epoch, len_train_datasets, len_test_datasets):
    metrics_dict = {
        "loss_train": [],
        "accuracy_train": [],
        "loss_eval": [],
        "accuracy_eval": [],
        "f1score_eval": []
    }
    # 训练模型
    for epoch_id in range(num_epoch):  # 进行训练循环
        loss_train, accuracy_train = train_batch(model, train_loader, optimizer, device, epoch_id, len_train_datasets)
        loss_eval, accuracy_eval, f1score_eval = eval(model, test_loader, device, len_test_datasets)
        metrics_dict["loss_train"].append(loss_train)
        metrics_dict["accuracy_train"].append(accuracy_train)
        metrics_dict["loss_eval"].append(loss_eval)
        metrics_dict["accuracy_eval"].append(accuracy_eval)
        metrics_dict["f1score_eval"].append(f1score_eval)

    return metrics_dict

def train_epoch_link(model, train_loader, test_datasets, optimizer, device, num_epoch, len_train_datasets, len_test_datasets):
    metrics_dict = {
        "loss_train": [],
        "accuracy_train": [],
        "ndcg_eval": [],
        "recall_eval": []
    }
    # 训练模型
    for epoch_id in range(num_epoch):  # 进行训练循环
        all_test_metrics = {}

        loss_train, accuracy_train = train_batch(model, train_loader, optimizer, device, epoch_id, len_train_datasets)
        for data_name, test_dataset in test_datasets.items():
            metrics_link = eval_link(model, test_dataset, data_name)
            all_test_metrics[data_name] = metrics_link
            print(f"{data_name} {metrics_link}")

        all_tstNum = sum([metrics['tstNum'] for metrics in all_test_metrics.values()])
        recall_eval = sum([metrics['Recall']*metrics['tstNum'] for metrics in all_test_metrics.values()]) / all_tstNum
        ndcg_eval = sum([metrics['NDCG']*metrics['tstNum'] for metrics in all_test_metrics.values()]) / all_tstNum
        metrics_dict["loss_train"].append(loss_train)
        metrics_dict["accuracy_train"].append(accuracy_train)
        metrics_dict["ndcg_eval"].append(recall_eval)
        metrics_dict["recall_eval"].append(ndcg_eval)

    return metrics_dict
