from pathlib import Path
import gc

import numpy as np
import pandas
import torch
from datasets import Dataset
from peft import TaskType, LoraConfig, get_peft_model, LoftQConfig
from sklearn.metrics import precision_score, accuracy_score, recall_score, jaccard_score, hamming_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, EvalPrediction
from accelerate import Accelerator
import os

from trl import SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_LINES = 15

# problems = ["A", "B", "C", "D", "E"]

if __name__ == "__main__":
    # for except_dataset in problems:
    accelerator = Accelerator()

    device_index = accelerator.process_index
    device_map = {"": device_index}

    # bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             # init_lora_weights="loftq",
                             # loftq_config=loftq_config,
                             target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                             r=8)

    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
                                                 torch_dtype=torch.bfloat16,
                                                 # quantization_config=bnb_config,
                                                 attn_implementation="flash_attention_2"
                                                 )
    # model.config.pad_token_id = model.config.eos_token_id  # todo only for phi2 and codellama?
    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    tokenizer.padding_side = 'right'

    # tokenizer.pad_token = tokenizer.eos_token  # todo only for phi2 and codellama?

    model.print_trainable_parameters()

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


    def prompt(instance):
        # return instance["incorrect_program"]
        correct = correct_programs[instance["instance"][11]]
        incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(instance["incorrect_program"].splitlines())))
        correction = "\n".join(instance["correction"])
        fl = " ".join(map(str, instance["fl"]))
        prompt = f"Reference implementation:\n{correct}\nStudent submission:\n{incorrect}\nIncorrect lines:\n{fl}\nCorrection:\n{correction}"
        return prompt


    df = pandas.read_feather("dataset_3.feather")
    ds = Dataset.from_pandas(df)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    ds = ds.map(lambda instance: {
        "text": prompt(instance),
        "problem": Path(instance["instance"]).stem.split("_")[0]}, batched=False)
    # ds = ds.filter(lambda instance: instance["problem"] != except_dataset)
    # ds = ds.map(lambda examples: tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt"), batched=True)
    # ds.set_format(type="torch", columns=["input_ids", "attention_mask"], device="cuda")

    # def compute_metrics(p: EvalPrediction):
    #     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #     result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    #     return result

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            # output_dir=f"./lora-gemma-2b-it-datasetv3-except-{except_dataset}-promptv2-targetmodules",
            output_dir=f"./lora-gen-gemma-2b-it-datasetv3-promptv2-targetmodules",
            learning_rate=1e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=500
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        dataset_text_field="text"
        # compute_metrics=compute_metrics
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
