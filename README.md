# LLM Basics

This is code I wrote for **Stanford CS336**: Language Modeling from Scratch! It was an amazing course, check it out at https://stanford-cs336.github.io/spring2025/. 

(Originally forked and slightly modified from their Assignment 1 repo: https://github.com/stanford-cs336/assignment1-basics. But this is self-contained.)

This repository includes implementations of the Transformer architecture, a BPETokenizer, a AdamOptimizer, w/ training scripts, and more. 
Using this code, we can train a small model (after lots of hyperparameter tuning!) for 1.5 hours on an H100 to get a validation loss of 3.55 on OpenWebText! 

## What's inside

#### `llm-basics/model/`
- Transformer model from torch primitives
- Custom Linear, Embedding, RMSNorm, MHSA, etc. modules
- Implements rotary positional embeddings, (causal) scaled multi-head self-attention.

#### `llm-basics/tokenizer/`
- Train a custom byte-pair encoding tokenizer (efficiently!)
- Tokenize a dataset

#### `llm-basics/train/`
- Implements AdamW, SGD optimizer from torch optimizer class
- Training loop with custom hyperparameters

#### `llm-basics/scripts/`
- Decode text from custom prompts, with temperature specifications
- Evaluate model on datasets
