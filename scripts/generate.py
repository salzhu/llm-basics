"""
Generate text from a model. 
"""

import numpy as np 
import torch 
import argparse
import random

from cs336_basics.model.transformer import TransformerLM
from cs336_basics.model.utils import softmax
from cs336_basics.tokenizer.bpe import BPETokenizer

"""
Function to decode from the model.
• Generates completions for a user-provided prompt (until an <|endoftext|> token).
• Allow the user to control the maximum number of generated tokens.
• Given a desired temperature value, apply softmax temperature scaling to the predicted next-word distributions before sampling
"""
def generate(model, tokenizer, prompt, temperature=1.0, max_tokens=None, end_token='<|endoftext|>'):
    encoded_prompt = tokenizer.encode(prompt)
    encoded_special_token = tokenizer.encode(end_token)[0]

    token_list = encoded_prompt.copy()

    while max_tokens is None or (max_tokens is not None and len(token_list) - len(encoded_prompt) < max_tokens):
        # generate new token 
        tokens = torch.Tensor([token_list[:model.context_length]]).int()
        logits = model.forward(tokens)
        probs = softmax(logits[0][-1], dim=0, temp=temperature)

        token = random.choices(np.arange(len(probs)), weights=probs)
        token_list.append(token[0])
        if token[0] == encoded_special_token:
            break

    return tokenizer.decode(token_list[len(encoded_prompt):])

"""
Main function to load tokenizer, model with state_dict, and decode and save generated text. 
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_tokens", type=int, default=256)

    parser.add_argument("--temp", type=float, default=0.2)

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--n_heads", type=int, default=8)

    parser.add_argument("--prompt", type=str, default='This is a prompt')

    parser.add_argument("--rope_theta", type=int, default=10000)

    parser.add_argument("--save_dir", type=str, default='runs')
    parser.add_argument("--model_name", type=str, requried=True)

    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--merges", type=str, required=True)

    args = parser.parse_args()


    transformer = TransformerLM(
        args.vocab_size, 
        args.context_length, 
        args.d_model, 
        args.n_layers, 
        args.n_heads, 
        args.d_ff, 
        args.rope_theta
    )

    tokenizer = BPETokenizer({}, {}, ["<|endoftext|>"])
    tokenizer.from_files(
        args.vocab,
        args.merges, 
        ["<|endoftext|>"]
    )

    saved_state_dict = torch.load(f'{args.save_dir}/{args.model_name}/final.pt')
    saved_state_dict = saved_state_dict['model']

    # Create a new state dict with modified keys
    new_state_dict = {}
    for key, value in saved_state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
        else:
            new_key = key
        new_state_dict[new_key] = value

    transformer.load_state_dict(new_state_dict)

    prompt = args.prompt

    print(generate(transformer, tokenizer, prompt, temperature=args.temp, max_tokens=args.max_tokens))

    