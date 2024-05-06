import argparse
import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas
import torch
from accelerate import Accelerator
from argparse_dataclass import ArgumentParser
from datasets import Dataset
from peft import TaskType, LoraConfig, get_peft_model
from sklearn.metrics import precision_score, accuracy_score, recall_score, jaccard_score, hamming_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, EvalPrediction

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
    max_program_length: int = 15
    prompt_version: int = 2
    dataset_version: int = 3
    seed: int = 42
    test_split: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 8
    epochs: int = 3

    use_8bit: bool = False

    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    use_target_modules: bool = True
    target_modules: list[str] = field(default_factory=lambda: ["k_proj", "o_proj", "q_proj", "v_proj"])

    cross_validate: bool = False
    excluded_benchmark: Optional[str] = None


def train(config: TrainConfig):
    accelerator = Accelerator()

    device_index = accelerator.process_index
    device_map = {"": device_index}

    bnb_config = BitsAndBytesConfig(load_in_8bit=config.use_8bit)
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                             target_modules=config.target_modules if config.use_target_modules else None,
                             modules_to_save=["score"],
                             r=config.lora_r,
                             lora_alpha=config.lora_alpha,
                             lora_dropout=config.lora_dropout)

    model = AutoModelForSequenceClassification.from_pretrained(config.base_model,
                                                               num_labels=config.max_program_length + 1,
                                                               torch_dtype=torch.bfloat16,
                                                               quantization_config=bnb_config,
                                                               problem_type="multi_label_classification",
                                                               attn_implementation="flash_attention_2",
                                                               device_map=device_map,
                                                               trust_remote_code=True)

    if config.base_model in NEEDS_PAD_TOKEN:
        model.config.pad_token_id = model.config.eos_token_id

    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    if config.base_model in NEEDS_PAD_LEFT:
        tokenizer.padding_side = 'left'

    if config.base_model in NEEDS_PAD_TOKEN:
        tokenizer.pad_token = tokenizer.eos_token

    model.print_trainable_parameters()

    def prompt(instance):
        if config.prompt_version == 1:
            return instance["incorrect_program"]

        elif config.prompt_version == 2:
            correct = correct_programs[instance["instance"][11]]
            incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(instance["incorrect_program"].splitlines())))
            prompt = f"<correct>{correct}\n<incorrect>{incorrect}"
            return prompt

        else:
            raise NotImplementedError()

    df = pandas.read_feather(f"dataset_{config.dataset_version}.feather")
    ds = Dataset.from_pandas(df)
    ds = ds.train_test_split(test_size=config.test_split, seed=config.seed)
    ds = ds.map(lambda instance: {
        "text": prompt(instance),
        "labels": [1 if instance["missing_lines"] else 0] + list(instance["line_scores"]) + [0] * max(0, config.max_program_length - len(instance["line_scores"])),
        "problem": Path(instance["instance"]).stem.split("_")[0]}, batched=False)
    if config.excluded_benchmark is not None:
        ds = ds.filter(lambda instance: instance["problem"] != config.excluded_benchmark)
    ds = ds.map(lambda examples: tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt"), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], device="cuda")

    def multi_label_metrics(predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
        recall = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
        jaccard = jaccard_score(y_true, y_pred, average='micro')
        hamming = hamming_loss(y_true, y_pred)
        # return as dictionary
        metrics = {'accuracy': accuracy,
                   'precision': precision,
                   'recall': recall,
                   'jaccard': jaccard,
                   'hamming': hamming}
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        return result

    model_name = config.base_model.split("/")[1]
    except_str = f"-except-{config.excluded_benchmark}" if config.excluded_benchmark is not None else ""
    targetmodules_str = "-targetmodules" if config.use_target_modules else ""

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"../models/{model_name}-datasetv{config.dataset_version}{except_str}-promptv{config.prompt_version}{targetmodules_str}",
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
        compute_metrics=compute_metrics
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
