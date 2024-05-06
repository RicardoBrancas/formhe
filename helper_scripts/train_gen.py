import argparse
import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas
import torch
from accelerate import Accelerator
from argparse_dataclass import ArgumentParser
from datasets import Dataset
from peft import TaskType, LoraConfig, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from transformers import TrainingArguments
from trl import SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NEEDS_PAD_TOKEN = {"microsoft/phi-2", "codellama/CodeLlama-7b-hf", "bigcode/starcoder2-3b", "bigcode/starcoder2-7b", "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"}
NEEDS_PAD_LEFT = {"bigcode/starcoder2-3b", "bigcode/starcoder2-7b"}

correct_programs = {
    "A": """color(1..k).
node(N) :- e(_, N).
node(N) :- e(N, _).
1 { assign(N,C) : color(C) } 1 :- node(N).
:- e(N,M), assign(N,C), assign(M,C).""",
    "B": """s(X) :- e(X,E).
k { sel(X) : s(X) } k.
inter(X, Y) :- e(X, E), e(Y, E), X != Y.
:- inter(X, Y), sel(X), sel(Y).""",
    "C": """v(X) :- e(X, _).
s(X) :- e(_, X).
0 { sel(X) : s(X) } k.
cov(X) :- v(X), e(X, S), sel(S).
:- not cov(X), v(X).""",
    "D": """vx(X) :- e(X,Y).
vx(X) :- e(Y,X).
0 { sel(X) : vx(X) } k.
:- not sel(X), not sel(Y), e(X,Y).""",
    "E": """1 { set(X, a) ; set (X, b) } 1 :- vertex(X).
:- edge(X, Y), set(X, S), set(Y, S)."""
}


@dataclass
class TrainConfig:
    base_model: str
    prompt_version: int = 1
    dataset_version: int = 3
    seed: int = 42
    test_split: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 4
    epochs: int = 3
    max_seq_length: int = 1024

    use_8bit: bool = False

    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    use_target_modules: bool = True
    target_modules: list[str] = field(metadata=dict(nargs='*', type=str), default_factory=lambda: ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])

    cross_validate: bool = False
    excluded_benchmark: Optional[str] = None


def train(config: TrainConfig):
    accelerator = Accelerator()

    device_index = accelerator.process_index
    device_map = {"": device_index}

    bnb_config = BitsAndBytesConfig(load_in_8bit=config.use_8bit)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             target_modules=config.target_modules if config.use_target_modules else None,
                             r=config.lora_r,
                             lora_alpha=config.lora_alpha,
                             lora_dropout=config.lora_dropout)

    model = AutoModelForCausalLM.from_pretrained(config.base_model,
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=bnb_config,
                                                 attn_implementation="flash_attention_2",
                                                 device_map=device_map,
                                                 trust_remote_code=True)

    if config.base_model in NEEDS_PAD_TOKEN:
        model.config.pad_token_id = model.config.eos_token_id

    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    tokenizer.padding_side = 'right'
    if config.base_model in NEEDS_PAD_LEFT:
        tokenizer.padding_side = 'left'

    if config.base_model in NEEDS_PAD_TOKEN:
        tokenizer.pad_token = tokenizer.eos_token

    model.print_trainable_parameters()

    def prompt(instance):
        if config.prompt_version == 1:
            correct = correct_programs[instance["instance"][11]]
            incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(instance["incorrect_program"].splitlines())))
            correction = "\n".join(instance["correction"])
            fl = " ".join(map(str, instance["fl"]))
            prompt = f"Reference implementation:\n{correct}\nStudent submission:\n{incorrect}\nIncorrect lines:\n{fl}\nCorrection:\n{correction}" + tokenizer.eos_token
            return prompt

        else:
            raise NotImplementedError()

    df = pandas.read_feather(f"dataset_{config.dataset_version}.feather")
    ds = Dataset.from_pandas(df)
    ds = ds.train_test_split(test_size=config.test_split, seed=config.seed)
    ds = ds.map(lambda instance: {
        "text": prompt(instance),
        "problem": Path(instance["instance"]).stem.split("_")[0]}, batched=False)
    if config.excluded_benchmark is not None:
        ds = ds.filter(lambda instance: instance["problem"] != config.excluded_benchmark)
    # ds = ds.map(lambda instance: tokenizer(instance["text"]), batched=True)

    model_name = config.base_model.split("/")[1]
    except_str = f"-except-{config.excluded_benchmark}" if config.excluded_benchmark is not None else ""
    targetmodules_str = "-targetmodules" if config.use_target_modules else ""

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"../models/{model_name}-repair-datasetv{config.dataset_version}{except_str}-promptv{config.prompt_version}{targetmodules_str}",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=max(1, int(4 / config.batch_size)),
            num_train_epochs=config.epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            bf16_full_eval=True,
            bf16=True
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        packing=True,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length
    )

    trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()

    del trainer
    del model
    del tokenizer
    del ds
    del df
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(TrainConfig, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config = parser.parse_args()

    if config.cross_validate:
        problems = ["A", "B", "C", "D", "E"]
        for problem in problems:
            config.excluded_benchmark = problem
            train(config)

    else:
        train(config)
