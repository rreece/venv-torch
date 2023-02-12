"""
Script for saving a checkpoint of a HuggingFace model.
"""


import os

MODEL_NAME = "textattack/bert-base-uncased-SST-2"
from transformers import BertForSequenceClassification as Model
from transformers import BertTokenizer as Tokenizer


def save_checkpoint(model, tokenizer, output_dir):
    """
    This will create output_dir containing:
    config.json
    pytorch_model.bin
    special_tokens_map.json
    tokenizer_config.json
    vocab.txt

    See:
    https://huggingface.co/transformers/v1.2.0/serialization.html#serialization-best-practices
    """
    assert not os.path.exists(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    model = Model.from_pretrained(MODEL_NAME)
    tokenizer = Tokenizer.from_pretrained(MODEL_NAME)
    output_dir = MODEL_NAME
    print("Saving checkpoint for %s" % output_dir) 
    save_checkpoint(model, tokenizer, output_dir)
    assert os.path.exists(output_dir)
    assert os.path.isdir(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
