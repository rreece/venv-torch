"""
Test Huggingface pretrained BERT.
"""


from hf_wrappers.models.bert import BertHandler


def test_inference():
    model = BertHandler()
    sample = "That dog is cute."
    result = model.run_inference(sample)
    assert result == 1
    sample = "That makes me sick."
    result = model.run_inference(sample)
    assert result == 0


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
