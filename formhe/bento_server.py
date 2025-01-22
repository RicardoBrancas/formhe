import base64
import gc
import os
import pickle
import signal
from dataclasses import dataclass
from typing import Optional

import bentoml

import formhe.utils.llm


@dataclass
class BentoConfig:
    fl_model_name: Optional[str] = None
    repair_model_name: Optional[str] = None
    fl_gpu_id: int = 0

    max_lines: int = 15
    repair_prompt_version: int = 3
    repair_gpu_id: int = 0
    repair_gpu_auto: bool = False
    repair_use_8bit: bool = False
    repair_use_4bit: bool = False
    repair_return_token_type_ids: bool = True


def predict(predictions):
    import torch
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    return probs[0].tolist()


@bentoml.service(traffic={"timeout": 600})
class FormHeLLM:
    def __init__(self) -> None:
        self.fl_model = None
        self.fl_tokenizer = None
        self.repair_model = None
        self.repair_tokenizer = None
        self.current_config: BentoConfig = None

    @bentoml.api()
    def load(self, config_base64: str):
        config = pickle.loads(base64.b64decode(config_base64.encode()))

        from peft import AutoPeftModelForSequenceClassification, AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, BitsAndBytesConfig
        import torch

        self.current_config = config

        if config.fl_model_name is not None:
            self.fl_model = AutoPeftModelForSequenceClassification.from_pretrained(f"../models/{config.fl_model_name}",
                                                                                   num_labels=config.max_lines + 1,
                                                                                   torch_dtype=torch.bfloat16,
                                                                                   quantization_config=None,
                                                                                   problem_type="multi_label_classification",
                                                                                   attn_implementation="flash_attention_2",
                                                                                   device_map={"": config.fl_gpu_id})
            self.fl_tokenizer = AutoTokenizer.from_pretrained(f"../models/{config.fl_model_name}",
                                                              device_map={"": config.fl_gpu_id})

        if config.repair_model_name is not None:
            if config.repair_use_8bit or config.repair_use_4bit:
                repair_bnb_config = BitsAndBytesConfig(load_in_8bit=config.repair_use_8bit,
                                                       load_in_4bit=config.repair_use_4bit,
                                                       bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
            else:
                repair_bnb_config = None

            if config.repair_gpu_auto:
                repair_device_map = "auto"
            else:
                repair_device_map = {"": config.repair_gpu_id}

            self.repair_model = AutoPeftModelForCausalLM.from_pretrained(f"../models/{config.repair_model_name}",
                                                                         torch_dtype=torch.bfloat16,
                                                                         quantization_config=repair_bnb_config,
                                                                         attn_implementation="flash_attention_2",
                                                                         device_map=repair_device_map)
            self.repair_tokenizer = AutoTokenizer.from_pretrained(f"../models/{config.repair_model_name}",
                                                                  device_map=repair_device_map)

    @bentoml.api
    def unload(self):
        import torch
        del self.fl_model
        self.fl_model = None
        del self.fl_tokenizer
        self.fl_tokenizer = None
        del self.repair_model
        self.repair_model = None
        del self.repair_tokenizer
        self.repair_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    @bentoml.api
    def exit(self):
        os.kill(os.getppid(), signal.SIGTERM)

    @bentoml.api
    def supports_fl(self) -> bool:
        return self.fl_model is not None

    @bentoml.api
    def fl(self, text: str) -> list:
        inputs = self.fl_tokenizer(text, return_tensors="pt").to(self.current_config.fl_gpu_id)
        output = self.fl_model(**inputs)
        return predict(output[0].cpu())

    @bentoml.api
    def supports_repair(self) -> bool:
        return self.repair_model is not None

    @bentoml.api
    def repair(self, text: str) -> str:
        inputs = self.repair_tokenizer(text, return_token_type_ids=self.current_config.repair_return_token_type_ids, return_tensors="pt").to(self.current_config.repair_gpu_id)
        output = self.repair_model.generate(**inputs, max_new_tokens=200)
        response = self.repair_tokenizer.decode(output[0], skip_special_tokens=False)
        return formhe.utils.llm.get_repair_response(response, self.current_config.repair_prompt_version, eos_token=self.repair_tokenizer.eos_token)
