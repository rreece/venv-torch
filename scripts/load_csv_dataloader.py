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


class SentimentDataset(Dataset):
    def __init__(self, csv_fn):
        # TODO: handle multiple files
        if isinstance(csv_fn, list):
            csv_fn = csv_fn[0]
        assert os.path.isfile(csv_fn)
        self.df = pd.read_csv(csv_fn)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Convert sentiment labels to numeric values
#        self.df['sentiment'] = self.df['sentiment'].map({'positive': 1, 'negative': 0})
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.df.iloc[idx]["content"]
        x = self.tokenizer(sample,
#                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                return_attention_mask=False,
                )
        print("DEBUG: keys = ", x.keys(), flush=True)
        input_ids = x["input_ids"]
        print("DEBUG: input_ids = ", input_ids, flush=True)
#        help(self.tokenizer)
        sentiment = self.df.iloc[idx]["label"]
        return {"sample": sample, "sentiment": sentiment}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles',  default=None, nargs='+',
            help='Input csv files.')
    return parser.parse_args()


def get_dataloader(fn):
    """
    csv to DataLoader
    """
    ds = SentimentDataset(fn)
    loader = DataLoader(ds,
                batch_size=2,
                num_workers=1,
#                pin_memory=True,
#                shuffle=True,
                )
    return loader


def main():
    args = parse_args()
    infiles = args.infiles
    print("DEBUG: ", infiles, flush=True)
    loader = get_dataloader(infiles)
    print(loader)
    for x in loader:
        print(x)
        assert False
    print("Done.")


if __name__ == "__main__":
    main()
