#!/usr/bin/env python3
"""
Script for loading CSV data and converting it to a HuggingFace dataset.

See also:
https://huggingface.co/docs/datasets/v1.0.0/torch_tensorflow.html
https://predictivehacks.com/?all-tips=how-to-load-csv-files-as-huggingface-dataset
https://wandb.ai/srishti-gureja-wandb/posts/How-To-Eliminate-the-Data-Processing-Bottleneck-With-PyTorch--VmlldzoyNDMxNzM1
"""


import argparse
import bisect
from itertools import islice
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


# DEBUG
DEBUG = True
if DEBUG:
    torch.set_printoptions(profile="full")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles',  nargs='+', default=None,
            help='Input csv files.')
    parser.add_argument('-m', '--max_batches', type=int, default=10,
            help='Max batches to process.')
    return parser.parse_args()


class SentimentDataset(Dataset):
    def __init__(self, csv_fn,
            tokenizer=None,
            max_length=512,
            content_name="text",
            label_name="label_text",
            return_attention_mask=True,
            return_token_type_ids=False):
        if isinstance(csv_fn, list):
            self._files = csv_fn
        else:
            self._files = [csv_fn]
        for f in self._files:
            assert os.path.isfile(f)
        row_counts = [self._count_rows(f) for f in self._files]
        self._cumulative = [0]
        for n in row_counts:
            self._cumulative.append(self._cumulative[-1] + n)
        self._cached_file_idx = None
        self._cached_df = None
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.content_name = content_name
        self.label_name = label_name
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids

    @staticmethod
    def _count_rows(path):
        with open(path) as f:
            return sum(1 for _ in f) - 1  # subtract header row

    def _get_row(self, idx):
        file_idx = bisect.bisect_right(self._cumulative, idx) - 1
        if file_idx != self._cached_file_idx:
            self._cached_df = pd.read_csv(self._files[file_idx])
            self._cached_file_idx = file_idx
        local_idx = idx - self._cumulative[file_idx]
        return self._cached_df.iloc[local_idx]

    def __len__(self):
        return self._cumulative[-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self._get_row(idx)
        sample = row[self.content_name]
        datum = dict()
        if self.tokenizer:
            tokenizer_outputs = self.tokenizer(sample,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=self.return_attention_mask,
                    return_token_type_ids=self.return_token_type_ids,
                    )
            datum["input_ids"] = tokenizer_outputs["input_ids"].squeeze()
            if self.return_attention_mask:
                datum["attention_mask"] = tokenizer_outputs["attention_mask"].squeeze()
        else:
            datum["sample"] = sample
        datum["label"] = row[self.label_name]
        return datum


def get_dataloader(fn):
    """
    Returns a DataLoader
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    ds = SentimentDataset(fn, tokenizer=tokenizer)
    dataloader = DataLoader(ds,
                batch_size=4,
                num_workers=0,
                shuffle=False,
                )
    return dataloader


def main():
    args = parse_args()
    infiles = args.infiles
    max_batches = args.max_batches
    dataloader = get_dataloader(infiles)
    print(dataloader)
    for i_batch, batch in tqdm(islice(enumerate(dataloader), max_batches), total=max_batches):
        if DEBUG:
            print("DEBUG: batch = ", batch)
        os.system("sleep 0.2s")
    print("Done.")


if __name__ == "__main__":
    main()
