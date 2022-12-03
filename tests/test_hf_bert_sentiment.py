"""
Test Huggingface pretrained BERT.
"""

import os
import torch

from hf_wrappers.models.bert import BertHandler


def test_inference():
    model = BertHandler()
    sample = "That dog is cute."
    result = model.run_inference(sample)
    assert result == 1

    sample = "That makes me sick."
    result = model.run_inference(sample)
    assert result == 0


def test_save_checkpoint():
    mh = BertHandler()
    mh.save_checkpoint(mh.model_name)

    output_dir = mh.model_name
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "config.json")
    output_vocab_file = os.path.join(output_dir, "vocab.txt")

    assert os.path.exists(output_model_file)
    assert os.path.exists(output_config_file)
    assert os.path.exists(output_vocab_file)


def main():
    model = BertHandler()
    
    print("Give a sample of text to score its sentiment. (q to quit)")
    
    while True:
        print("")
        sample = input("> ").strip()
        if sample.lower() == "q":
            break

        result = model.run_inference(sample)
        print(result)
    
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
