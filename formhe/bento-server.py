from dataclasses import dataclass
from typing import Optional

import bentoml
from argparse_dataclass import ArgumentParser


@dataclass
class BentoConfig:
    fl_model_name: str = "CodeLlama-7b-hf-datasetv3-promptv2"
    repair_model_name: Optional[str] = None

    max_lines: int = 15
    repair_prompt_version: int = 1

    gpu_id: int = 0


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
        from peft import AutoPeftModelForSequenceClassification
        from transformers import AutoTokenizer
        import torch
        self.fl_model = AutoPeftModelForSequenceClassification.from_pretrained(f"../models/{config.fl_model_name}",
                                                                               num_labels=config.max_lines + 1,
                                                                               torch_dtype=torch.bfloat16,
                                                                               problem_type="multi_label_classification",
                                                                               attn_implementation="flash_attention_2",
                                                                               device_map={"": config.gpu_id})
        # self.repair_model = AutoPeftModelForCausalLM.from_pretrained("../helper_scripts/lora-gen-gemma-2b-it-datasetv3-promptv2-targetmodules",
        #                                                              torch_dtype=torch.bfloat16,
        #                                                              attn_implementation="flash_attention_2", device_map={"": 0})
        self.tokenizer = AutoTokenizer.from_pretrained(f"../models/{config.fl_model_name}")

    @bentoml.api
    def fl(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        output = self.fl_model(**inputs)
        return predict(output[0].cpu())

    # @bentoml.api
    # def repair(self, text: str) -> str:
    #     inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
    #     output = self.repair_model.generate(**inputs, max_new_tokens=200)
    #     response = self.tokenizer.decode(output[0])
    #     if REPAIR_PROMPT_VERSION == 1:
    #         return response.split("Correction:\n", maxsplit=1)[1]
    #     else:
    #         raise NotImplementedError()
