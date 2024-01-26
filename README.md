# FormHe

Artifact link: https://doi.org/10.6084/m9.figshare.24297760

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

## Pre-set scripts

We include some pre-set scripts which can be used to replicate the results in our paper.  
We used the "Complete results (15 processes)" configuration.

### Complete results (15 processes) (estimated 40 hours)
- All instances
- All experiments
- Timeout of 10 minutes
- Memout of 60GB


    ./run_all_parallel.sh

### Complete results (1 process) (estimated 25 days)
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

Example: `python3 formhe/asp_integrated.py instances/A_0/0.lp`

## Batch Execution

The file `helper-scripts/evaluate.py` can be used to evaluate all instances and collect statistics. Using `--help` will show all options available.
Results will be stored in `analysis/data`.

Example: `python3 helper_scripts/evaluate.py run_identifier`