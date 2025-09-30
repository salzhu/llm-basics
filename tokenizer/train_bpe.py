"""
Trains a BPE tokenizer from a dataset (--path, --file). 
Accepts vocab_size as command-line argument. 
Saves the final vocabulary, merges performed, and pretoken index (to load for future purposes). 
"""

import pickle
import argparse
from cs336_basics.tokenizer.bpe import train as train_bpe

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/')
parser.add_argument('--file', type=str, default='TinyStoriesV2-GPT4-train')
parser.add_argument('--vocab_size', type=int, default=10000)
args = parser.parse_args()

if __name__ == '__main__':

    vocab, merges, index_to_list_count, pretokens_to_index = train_bpe(
        input_path=f'{args.path}{args.file}.txt',
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    with open(f'{args.file}_v{args.vocab_size}_vocab.pickle', 'wb') as file:
        pickle.dump(vocab, file)
    with open(f'{args.file}_v{args.vocab_size}_merges.pickle', 'wb') as file:
        pickle.dump(merges, file)
    with open(f'{args.file}_v{args.vocab_size}_index_to_list.pickle', 'wb') as file:
        pickle.dump(index_to_list_count, file)
    with open(f'{args.file}_v{args.vocab_size}_pretoken_index.pickle', 'wb') as file:
        pickle.dump(pretokens_to_index, file)