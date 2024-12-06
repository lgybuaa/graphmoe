import torch
import torch.nn.functional as F
from torch import autograd
from torch import nn, einsum, Tensor
from torch.nn import Module
from params import args


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
    return F.normalize(t, dim = - 1)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))