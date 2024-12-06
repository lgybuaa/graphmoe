import torch
import torch.nn.functional as F


def softmax_with_dropout(logits, dropout_rate=0.3):
    # 计算 softmax 概率分布
    probs = F.softmax(logits, dim=-1)

    # 生成一个掩码，丢弃一些概率
    mask = (torch.rand_like(probs) > dropout_rate).float()  # 随机生成掩码
    probs = probs * mask  # 逐元素相乘，丢弃一部分概率

    # 重新归一化，使概率总和为 1
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs


# 示例
logits = torch.randn(2, 5)  # 假设有两个样本，5个类别
print("原始logits:", logits)

# 计算带有随机丢弃的 softmax 输出
probs = softmax_with_dropout(logits, dropout_rate=0.3)
print("经过softmax后的概率（部分概率被置为零）:", probs)