"""
Handler using Huggingface pretrained BERT.

See:
https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification
"""


import torch
from transformers import BertTokenizer, BertForSequenceClassification


class BertHandler:

    def __init__(self, model_name=None):
        if model_name is None:
#            self.model_name = "textattack/bert-base-uncased-yelp-polarity"
            self.model_name = "textattack/bert-base-uncased-SST-2"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)

        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)

        self.model.eval()

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
