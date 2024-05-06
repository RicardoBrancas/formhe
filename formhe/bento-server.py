from dataclasses import dataclass
from typing import Optional

import bentoml
from argparse_dataclass import ArgumentParser


@dataclass
class BentoConfig:
    fl_model_name: Optional[str] = "CodeLlama-7b-hf-datasetv3-promptv2-targetmodules"
    repair_model_name: Optional[str] = None

    max_lines: int = 15
    repair_prompt_version: int = 1

    gpu_id: int = 0
    use_8bit: bool = False


parser = ArgumentParser(BentoConfig)
config, _ = parser.parse_known_args()


def predict(predictions):
    import torch
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    return probs[0].tolist()


@bentoml.service()
class FormHeLLM:
    def __init__(self) -> None:
        from peft import AutoPeftModelForSequenceClassification, AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, BitsAndBytesConfig
        import torch

        bnb_config = BitsAndBytesConfig(load_in_8bit=config.use_8bit)

        if config.fl_model_name is not None:
            self.fl_model = AutoPeftModelForSequenceClassification.from_pretrained(f"../models/{config.fl_model_name}",
                                                                                   num_labels=config.max_lines + 1,
                                                                                   torch_dtype=torch.bfloat16,
                                                                                   quantization_config=bnb_config,
                                                                                   problem_type="multi_label_classification",
                                                                                   attn_implementation="flash_attention_2",
                                                                                   device_map={"": config.gpu_id})
            self.fl_tokenizer = AutoTokenizer.from_pretrained(f"../models/{config.fl_model_name}")

        if config.repair_model_name is not None:
            self.repair_model = AutoPeftModelForCausalLM.from_pretrained(f"../models/{config.repair_model_name}",
                                                                         torch_dtype=torch.bfloat16,
                                                                         quantization_config=bnb_config,
                                                                         attn_implementation="flash_attention_2",
                                                                         device_map={"": config.gpu_id})
            self.repair_tokenizer = AutoTokenizer.from_pretrained(f"../models/{config.repair_model_name}")

    @bentoml.api
    def fl(self, text: str) -> list:
        inputs = self.fl_tokenizer(text, return_tensors="pt").to("cuda")
        output = self.fl_model(**inputs)
        return predict(output[0].cpu())

    @bentoml.api
    def repair(self, text: str) -> str:
        inputs = self.repair_tokenizer(text, return_tensors="pt").to("cuda")
        output = self.repair_model.generate(**inputs, max_new_tokens=200)
        response = self.repair_tokenizer.decode(output[0], skip_special_tokens=True)
        if config.repair_prompt_version == 1:
            return response.split("Correction:\n", maxsplit=1)[1]
        else:
            raise NotImplementedError()
