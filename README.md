# FormHe

## Repository Contents

- `formhe` contains the main source code for FormHe
- `helper-scripts` contains helper scripts used for evaluation
- `instances` contains the programs submitted by the students
- `correct-instances` and `mooshak` contain correct solutions for each problem
- `analysis` contains log files from FormHe executions and R code to analyze those executions
- `analysis/data` contains the log files

## Setup

The recommended environment is a based on a Python 3.10 install, along with the packages specified in `pyproject.toml`.  
You can install all dependencies by running:

    pip install .

### Setup for LLM-based FL

Using the LLM based fault localization requires a valid installation of `pytorch` with the packages `peft`, `flash-attn` and `bentoml`.

The LLM server must be launched before running FormHe using the following command in the `formhe/formhe` folder:

    python -m bentoml serve bento-server:FormHeLLM

The LLM server can be run in a different machine than Formhe.
If that is the case, you can point FormHe to the server using the following argument:

    python formhe/asp_integrated.py --llm-url=http://my-server-name:3000

FormHe will skip the LLM-based fault localization if the server does not respond after a set timeout.

## Pre-set scripts

We include some pre-set scripts which can be used to replicate the results in our paper.  
We used the "Complete results (15 processes)" configuration.

### Complete results (15 processes) (estimated 60 hours)
- All instances
- All experiments
- Timeout of 10 minutes
- Memout of 60GB


    ./run_all_parallel.sh

### Complete results (1 process) (estimated 40 days)
- All instances
- All experiments
- Timeout of 10 minutes
- Memout of 8GB


    ./run_all.sh

### Subset of results (1 process) (estimated 1 hour)
- 5 randomly selected submitted instances + 5 randomly selected synthetic instances
- Subset of experiments
- Timeout of 2 minutes
- Memout of 8GB


    ./run_subset.sh

## Execution
The main executable for FormHe is `formhe/asp_integrated.py`. Using `--help` will show all options available.

Example: `python3 formhe/asp_integrated.py instances/C/C_0_1/4.lp --skip-llm-fl`

## Batch Execution

The file `helper-scripts/evaluate.py` can be used to evaluate all instances and collect statistics. Using `--help` will show all options available.
Results will be stored in `analysis/data`.

Example: `python3 helper_scripts/evaluate.py run_identifier`