import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

torch.set_default_device("cuda")

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

self.model = AutoModelForSequenceClassification.from_pretrained(basepath / "gemma-2b-it", quantization_config=quantization_config, num_labels=20)
self.tokenizer = AutoTokenizer.from_pretrained(basepath / "gemma-2b-it")

chat = [
    {"role": "user", "content": text},
]
prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
outputs = self.model.generate(input_ids=inputs, max_new_tokens=512)
output_text = self.tokenizer.decode(outputs[0])
chat = reverse_gemma_chat_template(output_text)

