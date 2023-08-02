"""
Handler using Huggingface pretrained gpt2.

See:
https://huggingface.co/gpt2
https://huggingface.co/blog/how-to-generate
"""


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from hf_wrappers.models.handler import BaseHandler


class GPT2Handler(BaseHandler):

    def __init__(self, model_name=None):
        if model_name is None:
            model_name = "gpt2-small"
        super().__init__(model_name=model_name)

    def setup_model(self, model_name):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name,
                        pad_token_id=self.tokenizer.eos_token_id)
        self.max_new_tokens = 90

    def run_inference(self, sample):
        inputs = self.tokenizer([sample], return_tensors="pt")

        if self.device is not None:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.3,
                    )
            sequences = outputs.sequences
            result = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
            return result
