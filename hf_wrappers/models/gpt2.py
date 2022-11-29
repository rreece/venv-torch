"""
Handler using Huggingface pretrained gpt2.

See:
https://huggingface.co/gpt2
"""


from transformers import pipeline


class GPT2Handler:

    def __init__(self, model_name=None):
        if model_name is None:
            self.model_name = "gpt2"
        self.max_length = 240
        self.num_return_sequences = 1
        self.generator = pipeline("text-generation", model=self.model_name)

    def run_inference(self, sample):
        outputs = self.generator(sample,
                max_length=self.max_length,
                num_return_sequences=self.num_return_sequences)
        return outputs
