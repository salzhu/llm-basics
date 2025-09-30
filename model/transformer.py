import torch
import torch.nn as nn

from modules import Linear, Embedding, RMSNorm, SwiGLU
from attn import MultiheadSelfAttention

"""
Implementation of standard Transformer Block, with RMSNorm layernorms, SwiGLU, causal MHSA, and residual connections. 

d_model: (int) Dimensionality of the Transformer block inputs.
num_heads: (int) Number of heads to use in multi-head self-attention.
d_ff: (int) Dimensionality of the position-wise feed-forward inner layer.

max_seq_len: (int) Maximum sequence length that will be inputted. Required for ROPE
rope_theta: (float) value for the RoPE. 0 if no positional embedding
eps: (float = 1e-5) Epsilon value for numerical stability in layernorm

device: (torch.device | None = None) Device to store the parameters on
dtype: (torch.dtype | None = None) Data type of the parameters
"""
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, 
                 max_seq_len=2048, rope_theta=0, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, device=device, dtype=dtype)

    def forward(self, x):

        y = x + self.attn(self.ln1(x))
        z = y + self.ffn(self.ln2(y))

        return z
    
"""
Implementation of full Transformer module, with Embedding, Transformer blocks, residual stream, and linear output. 

vocab_size: (int) The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
context_length: (int) The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
d_model: (int) Dimensionality of the Transformer block inputs.
num_layers: (int) The number of Transformer blocks to use
num_heads: (int) Number of heads to use in multi-head self-attention.
d_model: (int) Dimensionality of the Transformer block feed-forward network.

rope_theta: (float) value for the RoPE. 0 if no positional embedding
eps: (float = 1e-5) Epsilon value for numerical stability in layernorm

device: (torch.device | None = None) Device to store the parameters on
dtype: (torch.dtype | None = None) Data type of the parameters
"""
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, 
                 rope_theta=0, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype))

        self.ln_final = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x):

        x = self.token_embeddings(x)

        for i in range(self.num_layers):
            x = self.layers[i](x)

        return self.lm_head(self.ln_final(x))