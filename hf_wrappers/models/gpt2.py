"""
Handler using Huggingface pretrained gpt2.

See:
https://huggingface.co/gpt2
https://huggingface.co/blog/how-to-generate
"""


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class GPT2Handler:

    def __init__(self, model_name=None):
        if model_name is None:
            self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name,
                        pad_token_id=self.tokenizer.eos_token_id)
        self.max_length = 240

        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)

        self.model.eval()

    def run_inference(self, sample):
        inputs = self.tokenizer([sample], return_tensors="pt")

        if self.device is not None:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=self.max_length)
            sequences = outputs.sequences
            result = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
            return result
