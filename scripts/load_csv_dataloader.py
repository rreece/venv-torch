#!/usr/bin/env python3
"""
Script for loading CSV data and converting it to a HuggingFace dataset.

See also:
https://huggingface.co/docs/datasets/v1.0.0/torch_tensorflow.html
https://predictivehacks.com/?all-tips=how-to-load-csv-files-as-huggingface-dataset
https://wandb.ai/srishti-gureja-wandb/posts/How-To-Eliminate-the-Data-Processing-Bottleneck-With-PyTorch--VmlldzoyNDMxNzM1
"""


import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# DEBUG
#torch.set_printoptions(profile="full")


class SentimentDataset(Dataset):
    def __init__(self, csv_fn,
            tokenizer=None,
            max_length=512,
            return_attention_mask=True):
        # TODO: handle multiple files
        if isinstance(csv_fn, list):
            csv_fn = csv_fn[0]
        assert os.path.isfile(csv_fn)
        self.df = pd.read_csv(csv_fn)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.return_attention_mask = return_attention_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.df.iloc[idx]["content"]
        datum = dict()
        if self.tokenizer:
            tokenizer_outputs = self.tokenizer(sample,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    return_attention_mask=self.return_attention_mask,
                    )
            datum["input_ids"] = tokenizer_outputs["input_ids"].squeeze()
            if self.return_attention_mask:
                datum["attention_mask"] = tokenizer_outputs["attention_mask"].squeeze()
        else:
            datum["sample"] = sample
        datum["label"] = self.df.iloc[idx]["label"]
        return datum



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles',  default=None, nargs='+',
            help='Input csv files.')
    return parser.parse_args()


def get_dataloader(fn):
    """
    csv to DataLoader
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
#    tokenizer = None
    ds = SentimentDataset(fn, tokenizer=tokenizer)
    dataloader = DataLoader(ds,
                batch_size=1,
                num_workers=0,
#                shuffle=True,
                )
    return dataloader


def main():
    args = parse_args()
    infiles = args.infiles
    dataloader = get_dataloader(infiles)
    print(dataloader)
    for x in dataloader:
        print(x)
        assert False
    print("Done.")


if __name__ == "__main__":
    main()
