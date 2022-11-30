"""
Test Huggingface pretrained GPT-2.
"""


from hf_wrappers.models.gpt2 import GPT2Handler


def test_inference():
    model = GPT2Handler()
    sample = "That dog is cute."
    result = model.run_inference(sample)
    generated_text = result[0]["generated_text"]
    assert len(generated_text) > 25


def main():
    model = GPT2Handler()
    
    print("Give a sample of text to prompt GPT2. (q to quit)")
    
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
