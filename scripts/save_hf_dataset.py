#!/usr/bin/env python3
"""
Script for saving a HuggingFace text dataset to csv.
"""


import argparse
import os

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name',
            help='Name of HuggingFace dataset to download.')
    return parser.parse_args()


def save_dataset(ds, output_dir):
    cwd = os.getcwd()
    assert not os.path.exists(output_dir)
    os.makedirs(output_dir)
    os.chdir(output_dir)
    print("Saving to CSVs...")
    for _split, _ds in ds.items():
        csv_fn = "%s.csv" % (_split)
        _ds.to_csv(csv_fn, index=None)
        print("Saved: ", csv_fn)
    os.chdir(cwd)


def main():
    args = parse_args()
    dataset_name = args.name
    output_dir = dataset_name
    print("Loading HuggingFace dataset...")
    ds = load_dataset(dataset_name)
    save_dataset(ds, output_dir)
    assert os.path.exists(output_dir)
    assert os.path.isdir(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
