"""
Handler using Huggingface pretrained t5.

See:
https://huggingface.co/docs/transformers/model_doc/t5
"""


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from hf_wrappers.models.handler import BaseHandler


class T5Handler(BaseHandler):

    def __init__(self, model_name=None):
        if model_name is None:
            model_name = "t5-base"
        super().__init__(model_name=model_name)

    def setup_model(self, model_name=None):
        self.model_name = model_name
        self.max_length = 512
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name,
                model_max_length=self.max_length)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def run_inference(self, sample):
        translate_en_fr = "translate English to French: "
        prepended_sample = translate_en_fr + sample
        inputs = self.tokenizer([prepended_sample], return_tensors="pt")
        input_ids = inputs.input_ids

        if self.device is not None:
            input_ids = input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids,
                    max_length=self.max_length)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
