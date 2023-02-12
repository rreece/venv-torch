"""
Wrapper for the amazon_polarity dataset at HuggingFace

See also:
https://huggingface.co/datasets/amazon_polarity
https://huggingface.co/docs/transformers/training#prepare-a-dataset
"""

from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer


def load(split="train"):
    dataset = load_dataset("amazon_polarity", split=split)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenize_f = partial(tokenize_function, tokenizer=tokenizer)
    tokenized_dataset = dataset.map(tokenize_f, batched=True)
    small_train_dataset = tokenized_dataset.shuffle(seed=42).select(range(1000))
    return small_train_dataset


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["content"], padding="max_length", truncation=True)
