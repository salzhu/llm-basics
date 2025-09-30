"""
Implementations of Torch / Transformer modules:
Linear, Embedding, RMSNorm, SwiGLU, SiLU, ROPE
"""

import torch
import torch.nn as nn
from einops import rearrange, einsum

from utils import silu, softmax

"""
Implementation of PyTorch Linear module, initialized with truncated Gaussian. 

in_features: (int) final dimension of the input
out_features: (int) final dimension of the output
device: (torch.device | None = None) Device to store the parameters on
dtype: (torch.dtype | None = None) Data type of the parameters
"""
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.randn((out_features, in_features), device=device, dtype=dtype), 
            std = 2/(in_features + out_features), 
            a = -3*2/(in_features + out_features), 
            b = 3*2/(in_features + out_features)
        ), requires_grad=True
        ) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight.T, x, "in_features out_features, ... in_features -> ... out_features")

"""
Implementation of PyTorch Embedding module.
Methods to set and get weights.  

num_embeddings: (int) Size of the vocabulary
embedding_dim: (int) Dimension of the embedding vectors, i.e., dmodel
device: (torch.device | None = None) Device to store the parameters on
dtype: (torch.dtype | None = None) Data type of the parameters
"""
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(
            torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype), 
            requires_grad=True)
        
    def set(self, weights):
        self.weight = torch.nn.Parameter(weights.to(device=self.device, dtype=self.dtype))

    def get(self):
        return self.weight
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

"""
Implementation of RMSNorm. (Forward pass upcasts to float32 first for precision purposes.)

d_model: (int) Hidden dimension of the model
eps: (float = 1e-5) Epsilon value for numerical stability
device: (torch.device | None = None) Device to store the parameters on
dtype: (torch.dtype | None = None) Data type of the parameters
"""
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.randn(d_model, device=device, dtype=dtype), requires_grad=True)

    def set(self, weights):
        self.weight = weights.to(device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.linalg.vector_norm(x, dim=-1)**2 / self.d_model + self.eps)
        result = einsum(self.weight, x / torch.unsqueeze(rms,-1), "d_model, batch_size sequence_length d_model -> batch_size sequence_length d_model")
    
        return result.to(in_dtype)
    
"""
Implementation of SwiGLU activation and feed-forward network. 

d_model: (int) Hidden dimension of the model
d_ff: (int) Feed-forward dimension of the model
device: (torch.device | None = None) Device to store the parameters on
dtype: (torch.dtype | None = None) Data type of the parameters
"""
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        result = silu(self.w1.forward(x))
        w3_output = self.w3.forward(x)
        result.mul_(w3_output)
        return self.w2.forward(result)
    
"""
Implementation of SiLU activation and feed-forward network. 

d_model: (int) Hidden dimension of the model
d_ff: (int) Feed-forward dimension of the model
device: (torch.device | None = None) Device to store the parameters on
dtype: (torch.dtype | None = None) Data type of the parameters
"""
class SiLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        result = silu(self.w1.forward(x))
        return self.w2.forward(result)

"""
Implementation of RotaryPositionalEmbedding. 

theta: (float) value for the RoPE
d_k: (int) dimension of query and key vectors
max_seq_len: (int) Maximum sequence length that will be inputted
device: (torch.device | None = None) Device to store the buffer on
"""
class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.register_buffer('cos_sin_matrix', torch.randn(max_seq_len, d_k // 2, 2), persistent=False)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                self.cos_sin_matrix[i,k,0] = torch.cos(torch.tensor(i / (theta ** (2 * k / d_k))))
                self.cos_sin_matrix[i,k,1] = torch.sin(torch.tensor(i / (theta ** (2 * k / d_k))))
        self.cos_sin_matrix.to(device=device)
        self.d_k = d_k

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        x_temp = rearrange(x, "... (d_k_split split2) -> ... d_k_split split2", d_k_split=self.d_k // 2, split2=2)

        result = torch.stack((
            self.cos_sin_matrix[token_positions,:,0] * x_temp[..., 0] - self.cos_sin_matrix[token_positions,:,1] * x_temp[..., 1], 
            self.cos_sin_matrix[token_positions,:,1] * x_temp[..., 0] + self.cos_sin_matrix[token_positions,:,0] * x_temp[..., 1]
        ), dim=-1)

        result = result.view(x.shape)

        return result

