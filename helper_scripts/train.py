import argparse
import gc
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
4
import numpy as np
import torch
from accelerate import Accelerator
from argparse_dataclass import ArgumentParser
from datasets import Dataset
from peft import TaskType, LoraConfig, get_peft_model, BOFTConfig
from sklearn.metrics import precision_score, accuracy_score, recall_score, jaccard_score, hamming_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, EvalPrediction

import formhe.utils.llm
from formhe.asp.problem import Problem
from formhe.utils.llm import fl_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NEEDS_PAD_TOKEN = {"microsoft/phi-2", "codellama/CodeLlama-7b-hf", "bigcode/starcoder2-3b", "bigcode/starcoder2-7b", "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"}
NEEDS_PAD_LEFT = {"bigcode/starcoder2-3b", "bigcode/starcoder2-7b"}


@dataclass
class TrainConfig:
    base_model: str
    max_program_length: int = 15
    prompt_version: int = 3
    dataset_version: int = 8
    seed: int = 42
    test_split: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 8
    epochs: int = 4
    gradient_checkpointing: bool = False
    device_auto: bool = False
    device_none: bool = False

    use_8bit: bool = False
    use_4bit: bool = False

    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05

    use_boft: bool = False
    boft_block_size: int = 8
    boft_n_butterfly_factor: int = 2
    boft_dropout: float = 0.05

    use_target_modules: bool = True
    target_modules: list[str] = field(metadata=dict(nargs='*', type=str), default_factory=lambda: ["q_proj", "v_proj"])

    cross_validate: bool = False
    excluded_benchmark: Optional[str] = None


def train(config: TrainConfig, problem_map: dict[str, Problem]):
    accelerator = Accelerator()

    if not config.device_auto and not config.device_none:
        device_index = accelerator.process_index
        device_map = {"": device_index}
    elif config.device_auto:
        device_map = "auto"
    else:
        device_map = None

    if config.use_8bit or config.use_4bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=config.use_8bit,
                                        load_in_4bit=config.use_4bit,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        bnb_config = None

    if config.use_lora and config.use_boft:
        raise ValueError("lora and boft are mutually exclusive")

    elif config.use_lora:
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                 target_modules=config.target_modules if config.use_target_modules else None,
                                 modules_to_save=["score"],
                                 r=config.lora_r,
                                 lora_alpha=config.lora_alpha,
                                 lora_dropout=config.lora_dropout)

    elif config.use_boft:
        peft_config = BOFTConfig(task_type=TaskType.SEQ_CLS,
                                 target_modules=config.target_modules if config.use_target_modules else None,
                                 modules_to_save=["score"],
                                 boft_dropout=config.boft_dropout)

    model = AutoModelForSequenceClassification.from_pretrained(config.base_model,
                                                               num_labels=config.max_program_length + 1,
                                                               torch_dtype=torch.bfloat16,
                                                               quantization_config=bnb_config,
                                                               problem_type="multi_label_classification",
                                                               attn_implementation="flash_attention_2",
                                                               device_map=device_map,
                                                               trust_remote_code=False)

    if config.base_model in NEEDS_PAD_TOKEN:
        model.config.pad_token_id = model.config.eos_token_id

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    tokenizer.add_tokens(formhe.utils.llm.SPECIAL_TOKENS, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if config.base_model in NEEDS_PAD_LEFT:
        tokenizer.padding_side = 'left'

    if config.base_model in NEEDS_PAD_TOKEN:
        tokenizer.pad_token = tokenizer.eos_token

    model.print_trainable_parameters()

    ds = Dataset.load_from_disk(f"datasets/dataset_{config.dataset_version}")
    ds = ds.train_test_split(test_size=config.test_split, seed=config.seed)
    ds = ds.map(lambda instance: {
        "text": fl_prompt(version=config.prompt_version, title=problem_map[instance["problem"]].title, **instance),
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
    peft_str = "lora" if config.use_lora else "boft"
    quantization_str = "-8bit" if config.use_8bit else ("-4bit" if config.use_4bit else "")
    except_str = f"-except-{config.excluded_benchmark}" if config.excluded_benchmark is not None else ""
    targetmodules_str = "-targetmodules" if config.use_target_modules else ""

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"./models/{model_name}-{peft_str}{quantization_str}-datasetv{config.dataset_version}{except_str}-promptv{config.prompt_version}{targetmodules_str}",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=max(1, int(8 / config.batch_size)),
            num_train_epochs=config.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            # bf16_full_eval=True,
            bf16=True,
            # load_best_model_at_end=True,
            gradient_checkpointing=config.gradient_checkpointing,
            metric_for_best_model="accuracy",
            optim="adafactor",
            gradient_checkpointing_kwargs={"use_reentrant": False},  # must be false for DDP
            ddp_find_unused_parameters=False  # if use DDP is false, otherwise true
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
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(TrainConfig, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config = parser.parse_args()

    problems = [Problem.from_yaml_file(problem_file) for problem_file in glob.glob("problems/*.yaml")]
    problem_map = {problem.name: problem for problem in problems}

    if config.cross_validate:
        for problem in problems:
            config.excluded_benchmark = problem.name
            train(config, problem_map)

    else:
        train(config, problem_map)
