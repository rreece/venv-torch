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

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data',
            help='Path to dataset directory.')
    return parser.parse_args()


def load_csv_dataset(data_files):
    for k, v in data_files.items():
        assert os.path.exists(v)
        assert os.path.isfile(v)
    ds = load_dataset("csv", data_files=data_files)
    return ds


def do_work(ds):
    """
    TODO
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    dataset = dataset.map(lambda e: tokenizer(e['sentence1']), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)


def main():
    args = parse_args()
    data_path = args.data
    data_files = {
        "train": os.path.join(data_path, "train.csv"),
        "test": os.path.join(data_path, "test.csv"),
    }
    ds = load_csv_dataset(data_files)
    print(ds)
    print("Done.")


if __name__ == "__main__":
    main()
