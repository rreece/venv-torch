"""
Handler base class for wrapping HuggingFace models.
"""


import torch


class BaseHandler:

    def __init__(self,
            model_name=None,
            device_name=None,
            mode=None):
        self.setup_model(model_name=model_name)
        self.setup_device(device_name=device_name)
        self.setup_mode(mode=mode)

    def setup_model(self, model_name=None):
        """
        Implement this in a derived class.
        """
        self.model_name = model_name

    def setup_device(self, device_name=None):
        if device_name is None:
            if torch.cuda.is_available():
                device_name = "cuda"

        if device_name is not None:
            self.device = torch.device(device_name)
            self.model.to(self.device)
        else:
            self.device = None

    def setup_mode(self, mode=None):
        if mode is None:
            mode = "eval"
        if mode == "eval":
            self.model.eval()
        if mode == "train":
            self.model.train()

    def run_inference(self, sample):
        """
        Implement this in a derived class.
        """
        inputs = self.tokenizer(sample, return_tensors="pt")

        if self.device is not None:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class_id = probs.argmax().item()
            return predicted_class_id

    def save_checkpoint(self, output_dir=None):
        """
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        output_config_file = os.path.join(output_dir, "config.json")
        output_vocab_file = os.path.join(output_dir, "vocab.txt")
        os.makedirs(output_dir)
        torch.save(self.model.state_dict(), output_model_file)
        self.model.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(output_dir)

        See:
        https://huggingface.co/transformers/v1.2.0/serialization.html#serialization-best-practices
        """
        if output_dir is None:
            output_dir = self.model_name
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
