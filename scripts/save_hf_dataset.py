#!/usr/bin/env python3
"""
Script for saving a HuggingFace text dataset to csv.

See also:

-	https://huggingface.co/docs/datasets/loading
-	https://datascience.stackexchange.com/questions/35868/how-to-store-strings-in-csv-with-new-line-characters
"""


import argparse
import os

from datasets import load_dataset, DatasetDict


EOL = '\n'
EOL_ENCODED = '\N{LATIN SMALL LETTER THORN}'
CHUNK_SIZE = 10000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--encode_eol",
            action="store_true", default=False,
            help="Whether to swap EOL characters with an encoded version.")
    parser.add_argument("-p", "--path",
            help="Name of HuggingFace dataset to download.")
    parser.add_argument("-n", "--name",
            default=None,
            help="Optional second name of HuggingFace dataset to download; the 2nd argument to load_dataset.")
    parser.add_argument("-s", "--split",
            default=None,
            help="Name of split within the dataset.")
    parser.add_argument("-l", "--language",
            default=None)
    parser.add_argument("-d", "--date",
            default=None)
    return parser.parse_args()


def encode_eol(sample, content_name="text"):
    sample[content_name] = sample[content_name].replace(EOL, EOL_ENCODED)
    return sample


def save_dataset(ds, output_dir, filename="data", do_encode_eol=False):
    cwd = os.getcwd()
    assert not os.path.exists(output_dir)
    os.makedirs(output_dir)
    os.chdir(output_dir)
    print(ds)
    print("Saving to CSVs...")
    if do_encode_eol:
        ds = ds.map(encode_eol)
    n = len(ds)
    for i, start in enumerate(range(0, n, CHUNK_SIZE)):
        end = min(start + CHUNK_SIZE, n)
        chunk_df = ds.select(range(start, end)).to_pandas()
        chunk_df.index = range(start, end)
        fn = "%s_%03i.csv" % (filename, i)
        chunk_df.to_csv(fn, index=True, index_label="index")
    print("Saved: ", output_dir)
    os.chdir(cwd)


def main():
    args = parse_args()
    dataset_name = args.path
    output_dir = os.path.join("out", dataset_name)
    print("Loading HuggingFace dataset...")
    ds_args = {"path": dataset_name}
    if args.name:
        ds_args["name"] = args.name
        output_dir = os.path.join(output_dir, args.name)
    if args.split:
        ds_args["split"] = args.split
    if args.language:
        ds_args["language"] = args.language
    if args.date:
        ds_args["date"] = args.date
    print("DEBUG: ds_args = ", ds_args)
    ds = load_dataset(**ds_args)
    ds_key = None
    if isinstance(ds, DatasetDict):
        ds_keys = list(ds.keys())
        print("This is a DatasetDict with keys: ", ds_keys)
        ds_key = ds_keys[0]
        print("Using first key: ", ds_key)
        ds = ds[ds_key]
    filename = args.split or ds_key or dataset_name
    save_dataset(ds, output_dir, filename=filename, do_encode_eol=args.encode_eol)
    assert os.path.exists(output_dir)
    assert os.path.isdir(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
