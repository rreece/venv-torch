"""
Test Huggingface pretrained t5.
"""


from hf_wrappers.models.t5 import T5Handler


def test_inference():
    t5 = T5Handler()
    sample = "I would like a bottle of red wine, please."
    answer = "Je voudrais une bouteille de vin rouge, s'il vous plaît."
    result = t5.run_inference(sample)
    assert result == answer


def main():
    t5 = T5Handler()
    
    print("Give a sample of text to prompt T5. (q to quit)")
    
    while True:
        print("")
        sample = input("> ").strip()
        if sample.lower() == "q":
            break

        result = t5.run_inference(sample)
        print(result)

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
