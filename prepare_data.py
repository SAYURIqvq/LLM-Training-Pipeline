"""
Data preparation: tokenize a HuggingFace dataset into a .pt file for pre-training.

Usage:
    python prepare_data.py --dataset openwebtext --tokenizer gpt2 --output data/train.pt
    python prepare_data.py --dataset cerebras/SlimPajama-627B --split train --output data/slimpajama.pt
"""

import argparse
import os

import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def tokenize_with_tiktoken(dataset, encoding_name: str, text_key: str, max_tokens: int = None):
    """Tokenize dataset using tiktoken (fast BPE)."""
    enc = tiktoken.get_encoding(encoding_name)
    all_tokens = []
    total = 0

    for example in tqdm(dataset, desc="Tokenizing"):
        text = example[text_key]
        if not text:
            continue
        tokens = enc.encode_ordinary(text)
        all_tokens.extend(tokens)
        total += len(tokens)

        if max_tokens and total >= max_tokens:
            all_tokens = all_tokens[:max_tokens]
            break

    return torch.tensor(all_tokens, dtype=torch.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        choices=["gpt2", "cl100k_base", "o200k_base"])
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max tokens to process (None = all)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--streaming", action="store_true", default=False,
                        help="Use streaming mode for large datasets")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        streaming=args.streaming,
    )

    print(f"Tokenizing with {args.tokenizer}...")
    tokens = tokenize_with_tiktoken(ds, args.tokenizer, args.text_key, args.max_tokens)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"input_ids": tokens}, args.output)
    print(f"Saved {len(tokens):,} tokens to {args.output}")
    print(f"Vocab: {args.tokenizer} | Size: {tokens.nbytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
