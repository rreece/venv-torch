"""
Handler using Huggingface pretrained BERT.

See:
https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification
"""


import torch
from transformers import BertTokenizer, BertForSequenceClassification

from hf_wrappers.models.handler import BaseHandler


class BertHandler(BaseHandler):

    def __init__(self, model_name=None):
        if model_name is None:
            model_name = "textattack/bert-base-uncased-SST-2"
        super().__init__(model_name=model_name)

    def setup_model(self, model_name=None):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)

    def run_inference(self, sample):
        inputs = self.tokenizer(sample, return_tensors="pt")

        if self.device is not None:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class_id = probs.argmax().item()
            return predicted_class_id
