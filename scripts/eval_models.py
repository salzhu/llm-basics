"""
Evaluates loss on a dataset of a model. 
Main function that evaluates and records a list of models. 
"""

import torch 
from tqdm import tqdm 
import argparse
import numpy as np 

from cs336_basics.train.utils import CrossEntropyLoss
from cs336_basics.train.utils import load_batch
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.train.optimizer import AdamW

"""
Evaluates loss on a dataset of a model. 
"""
def eval(args, dataset, model, device):
    loss_fn = CrossEntropyLoss()

    total_loss = 0 

    for _ in tqdm(range(args.n_batches)):

        val_inputs, val_targets = load_batch(dataset, args.batch_size, args.context_length, device)
                
        val_outputs = model(val_inputs)
        val_outputs = val_outputs.view(-1, val_outputs.size(-1))
        val_targets = val_targets.view(-1)
        
        with torch.no_grad():
            val_loss = loss_fn(val_outputs, val_targets)
            total_loss += val_loss.cpu().item()

    return total_loss / args.n_batches

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesValid_tokenized.npy')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--n_batches", type=int, default=1000)

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=16)

    parser.add_argument("--rope_theta", type=int, default=10000)

    # training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--save_dir", type=str, default='runs')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = np.lib.format.open_memmap(args.dataset, mode='r').astype(int)

    model_paths = [
        'base_1e3_128_l6_d768',
        'base_1e3_64_l6_d768',
        'base_2e3_128_l6_d768',
        'base_5e4_128_l6_d768',
    ]

    transformer = TransformerLM(
        args.vocab_size, 
        args.context_length, 
        args.d_model, 
        args.n_layers, 
        args.n_heads, 
        args.d_ff, 
        args.rope_theta, 
        device=device
    )
    opt = AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon,
    )
    transformer.to(device)

    for model_name in model_paths: 
        # Load the saved state dict
        saved_state_dict = torch.load(f'{args.save_dir}/{model_name}/final.pt')
        saved_state_dict = saved_state_dict['model']

        # Create a new state dict with modified keys
        new_state_dict = {}
        for key, value in saved_state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Load the modified state dict into your model
        transformer.load_state_dict(new_state_dict)
        # load_checkpoint(f'{args.save_dir}/{model_name}/final.pt', transformer, opt)
        eval_loss = eval(args, dataset, transformer, device)
        print(model_name, eval_loss)