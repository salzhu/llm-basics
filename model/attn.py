import torch 
import numpy as np 
import torch.nn as nn 

from einops import rearrange, einsum

from modules import Linear, ROPE

"""
Implementation of standard scaled dot product attention, with attention masking. 

Q: (Tensor) Q (query) attention matrix, shape (n, d_k)
K: (Tensor) K (key) attention matrix, shape (m, d_k)
V: (Tensor) V (value) attention matrix, shape (m, d_v)
mask: (boolean Tensor) optional user-provided boolean mask of shape (seq_len,seq_len)
"""
def scaled_dot_product_attention(Q, K, V, mask):

    result = einsum(Q, K, "batch_size ... seq_len_1 d_k, batch_size ... seq_len_2 d_k -> batch_size ... seq_len_1 seq_len_2")
    result = result / np.sqrt(K.shape[-1])

    if mask is not None:
        result[...,~mask] = -torch.inf

    result = torch.nn.functional.softmax(result, dim=-1)
    result = einsum(result, V, "batch_size ... seq_len_1 seq_len_2, batch_size ... seq_len_2 d_v -> batch_size ... seq_len_1 d_v")

    return result

"""
Implementation of causal multi-head self attention. 

d_model: (int) Dimensionality of the Transformer block inputs.
num_heads: (int) Number of heads to use in multi-head self-attention.
max_seq_len: (int) Maximum sequence length that will be inputted. Required for ROPE
rope_theta: (float) value for the RoPE. 0 if no positional embedding
device: (torch.device | None = None) Device to store the parameters on
dtype: (torch.dtype | None = None) Data type of the parameters
"""
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048, rope_theta=0, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads 

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.use_rope = False

        if rope_theta != 0:
            self.rope = ROPE(rope_theta, self.d_k, max_seq_len, device=device)
            self.use_rope = True

    def forward(self, x, token_positions=None):

        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).to(torch.bool)
        mask = ~mask

        q_x = self.q_proj.forward(x)
        k_x = self.k_proj.forward(x)
        v_x = self.v_proj.forward(x)

        # move sequence length to second to last dim to process multihead

        q_x = rearrange(q_x, "... (n_head d_k) -> ... n_head d_k", n_head=self.num_heads, d_k=self.d_k)
        k_x = rearrange(k_x, "... (n_head d_k) -> ... n_head d_k", n_head=self.num_heads, d_k=self.d_k)
        v_x = rearrange(v_x, "... (n_head d_k) -> ... n_head d_k", n_head=self.num_heads, d_k=self.d_v)

        q_x = rearrange(q_x, "... seq_len n_head d_k -> ... n_head seq_len d_k")
        k_x = rearrange(k_x, "... seq_len n_head d_k -> ... n_head seq_len d_k")
        v_x = rearrange(v_x, "... seq_len n_head d_k -> ... n_head seq_len d_k")

        if token_positions is None:
            token_positions = torch.arange(q_x.shape[-2])

        if self.use_rope:
            q_x = self.rope(q_x, token_positions)
            k_x = self.rope(k_x, token_positions)

        result = scaled_dot_product_attention(q_x, k_x, v_x, mask=mask)
        result = rearrange(result, "... n_head seq_len d_k -> ... seq_len (n_head d_k)")
        result = self.output_proj.forward(result)
        
        return result