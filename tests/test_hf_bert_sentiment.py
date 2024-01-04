"""
Test Huggingface pretrained BERT.
"""

import os
import shutil

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
    # save checkpoint
    bert = BertHandler()
    checkpoint_name = bert.model_name + "_test"
    bert.save_checkpoint(checkpoint_name)

    # check that it exists
    output_dir = checkpoint_name
    output_model_file = os.path.join(output_dir, "model.safetensors")
    output_config_file = os.path.join(output_dir, "config.json")
    output_vocab_file = os.path.join(output_dir, "vocab.txt")
    assert os.path.exists(output_model_file)
    assert os.path.exists(output_config_file)
    assert os.path.exists(output_vocab_file)

    # load from saved checkpoint
    del bert
    bert = BertHandler(checkpoint_name)
    assert bert
    del bert

    # rm the saved checkpoint
    assert os.path.exists(output_dir)
    assert os.path.isdir(output_dir)
    shutil.rmtree(output_dir)
    assert not os.path.exists(output_dir)


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
