import numpy as np 
import torch 
import torch.nn as nn
from einops import rearrange, einsum
from numpy.lib.stride_tricks import sliding_window_view

"""
Cross entropy loss as a module. 
Subtracts the largest element for numerical stability. Cancels out log and exp whenever possible.
"""
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):

        inputs -= torch.unsqueeze(torch.max(inputs, dim=1).values,1)
        loss = torch.log(torch.sum(torch.exp(inputs),dim=-1))
        loss -= inputs[np.arange(len(inputs)), targets]

        return torch.mean(loss)

"""
Cosine learning rate schedule with warmup

t: (int) Current training step 
a_max: (float) Maximum learning rate
a_min: (float) Minimum learning rate 
T_w: (int) number of warm-up iterations 
T_c: (int) number of cool-down iterations 
"""
def learning_rate_schedule(t, a_max, a_min, T_w, T_c):
    if t < T_w:
        return t / T_w * a_max 
    elif t <= T_c:
        return a_min + 0.5 * (1 + np.cos((t - T_w) / (T_c - T_w) * np.pi)) * (a_max - a_min)
    else:
        return a_min 
    
"""
Implements gradient clipping
"""
def gradient_clipping(params, max_l2_norm, eps=1e-6):
    grads = []
    for param in params:
        if param.grad is not None:
            grads.append(param.grad.view(-1))

    all_grads = torch.cat(grads, dim=0)
    norm = torch.norm(all_grads)

    if norm >= max_l2_norm:
        for param in params:
            if param.grad is not None:
                param.grad *= max_l2_norm / (norm + eps)

def load_batch(
        x: np.ndarray, 
        batch_size: int, 
        context_length: int, 
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = np.random.randint(0, len(x)-context_length, size=batch_size)
    all_windows = sliding_window_view(x, context_length + 1)
    data = all_windows[start_indices]
    input_sequences = torch.tensor(data[:, :-1], dtype=torch.long, device=device)
    target_sequences = torch.tensor(data[:, 1:], dtype=torch.long, device=device)
    return input_sequences, target_sequences

def data_loading(dataset, batch_size, context_length, device):
    inputs = []
    targets = []
    possible_start = np.arange(0, len(dataset) - context_length)
    for i in range(batch_size):
        ind = np.random.choice(possible_start) 
        inputs.append(dataset[ind:ind + context_length])
        targets.append(dataset[ind + 1:ind + 1 + context_length])

    return torch.tensor(np.array(inputs),device=device), torch.tensor(np.array(targets),device=device)

def save_checkpoint(model, optimizer, iteration, out):
    obj = {}
    obj['model'] = model.state_dict()
    obj['optimizer'] = optimizer.state_dict()
    obj['iteration'] = iteration 
    torch.save(obj, out)
    return 

def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']