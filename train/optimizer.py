from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
 
"""
AdamW optimizer as a subclass of torch.optim.Optimizer. 

params: (torch.nn.Module) Learnable parameters to update
lr: (float) Learning rate
betas: ((float, float)) Hyperparameters for controlling first and second moment updates
weight_decay: (float) Weight decay term for regularization
eps: (float) For numerical stability 
"""
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "lambda": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lmbda = group["lambda"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = beta1 * m + (1 - beta1) * grad 
                v = beta2 * v + (1 - beta2) * grad * grad

                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps) 
                p.data -= lr * lmbda * p.data
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
                del t, m, v, grad

        return loss

"""
Custom implementation of SGD optimizer from CS336 staff. 
"""
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss
   

"""
Example training loop using optimizer. 
"""
if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1000)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step