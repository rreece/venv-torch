"""
Test Huggingface pretrained GPT-2.
"""


from hf_wrappers.models.gpt2 import GPT2Handler


def test_inference():
    gpt2 = GPT2Handler()
    sample = "That dog is cute."
    result = gpt2.run_inference(sample)
    assert len(result) > 25


def main():
    gpt2 = GPT2Handler()
    
    print("Give a sample of text to prompt GPT2. (q to quit)")
    
    while True:
        print("")
        sample = input("> ").strip()
        if sample.lower() == "q":
            break

        result = gpt2.run_inference(sample)
        print("")
        print(result)

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
