# FormHe

## Repository Contents

- `formhe` contains the main source code for FormHe
- `helper-scripts` contains helper scripts used for evaluation and machine learning tasks
- `instances` contains incorrect programs used for testing and evaluation. `instances/mooshak` contains real instances submitted by student while the rest are synthetic.
- `correct-instances` and `problems` contains correct solutions and info for each problem
- `analysis` contains log files from FormHe executions and R code to analyze those executions
- `analysis/data` contains the log files

## Setup

The recommended environment is a based on a Python 3.10 install, along with the packages specified in `pyproject.toml`.  
You can install all dependencies by running the following command. You can also optionally create a virtual environment or conda environment first if you wish.

    pip install .

### Setup for LLM-based FL and Repair

Using the LLM methods requires a valid installation of `pytorch` with the packages `peft`, `flash-attn`, `bentoml` and `bitsandbytes`.

The LLM finetuned weights must be downloaded from [https://figshare.com/s/0ceb022e695efcb0d624](https://figshare.com/s/0ceb022e695efcb0d624) and placed on the `models` folder before running the LLM experiments.

The LLM server must be launched before running FormHe using the following command in the `./formhe` folder:

    PYTHONPATH=.. python -m bentoml serve bento_server:FormHeLLM

The LLM server can be run in a different machine than Formhe.
If that is the case, you can point FormHe to the server using the following argument:

    python formhe/asp_integrated.py --llm-url=http://my-server-name:3000

FormHe will skip the LLM-based fault localization if the server does not respond after a set timeout.

## Pre-set scripts

We include some pre-set scripts which can be used to replicate the results in our paper. 

These scripts include some important parameters which might need to be changed depending on the execution environment:

- The number of parallel processes used can be configured by changing the `--p=20` parameter. We used 20 processes in our experiments.
- For experiments using LLMs, the LLM server can be executed in a different machine (as described above). If so, please change the `--llm-url=http://localhost:3000` and `--url http://localhost:3000` parameters to point to the right host and port.
- The FL model and repair model might not both fit in a single GPU. By default, we load the FL model into GPU 0 and the repair model into GPU 1. This can be changed with the `--fl-gpu-id 0` and `--repair-gpu-id 1` parameters.

### Non LLM results (using 20 processes)
- All instances
- Experiments that do not use LLMs
- Timeout of 10 minutes
- Memout of 60GB


    ./run_no_llm.sh

### LLM Results (20 process)
- All instances
- Experiments that use the default LLMs (Gemma FL and CodeGemma Repair)
- Timeout of 10 minutes
- Memout of 60GB


    ./run_with_default_llms.sh

## Execution
The main executable for FormHe is `formhe/asp_integrated.py`. Using `--help` will show all options available.

Example: `python3 formhe/asp_integrated.py instances/mooshak/01/A_1_1.lp --skip-llm-fl --skip-llm-repair`

## Batch Execution

The file `helper-scripts/evaluate.py` can be used to evaluate all instances and collect statistics. Using `--help` will show all options available.
Results will be stored in `analysis/data`.

Example: `python3 helper_scripts/evaluate.py run_identifier`

## Replication of data and plots used in the paper

The file `analysis/analysis.R` can be used to replicate all of the data and plots used in the paper. Plots will be stored in the `analysis/plots` folder.
You might need to install the R packages listed at the beginning of the file, depending on your R installation.

## Machine learning scripts

The `helper_scipts/dataset_gen.py` script was used to generate the syntehtic data used in the paper, while the `helper_scipts/train.py` and `helper_scipts/train_gen.py` scripts were used to finetune the Fault Localization and Repair models, respectively.