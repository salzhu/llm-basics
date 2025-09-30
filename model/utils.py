import torch 

"""
Computes Swish activation function (SiLU), x * \sigmoid(x).
"""
def silu(x):
    return x * torch.sigmoid(x)

"""
Applies softmax operation on a tensor. Subtracts maximum value along dim for numerical stability. 

v: (Tensor) Tensor to apply softmax
dim: (int) Dimension along which to apply the softmax
temp: (float) Exponent in softmax sum
"""
def softmax(v, dim, temp=1.0):

    v /= temp
    v = torch.movedim(v, dim, 0)
    v -= torch.amax(v, dim=0) 

    return torch.movedim(torch.exp(v) / torch.sum(torch.exp(v), 0), 0, dim)