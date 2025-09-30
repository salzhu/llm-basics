"""
Tokenize a dataset using trained BPETokenizer. Loads from .txt and saves to .npy. 
"""

import numpy as np
from cs336_basics.tokenizer.bpe import BPETokenizer

if __name__ == '__main__':

    bpe2 = BPETokenizer(vocab={}, merges={}, special_tokens=["<|endoftext|>"])

    encoded = bpe2.encode_from_pretokens(
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/owt_train.txt',
        ["<|endoftext|>"]
    )

    results = np.array(encoded, dtype=np.uint16)
    np.save('OWTTrain_tokenized.npy', results)
