# FormHe

## Repository Contents

- `formhe` contains the main source code for FormHe
- `helper-scripts` contains helper scripts used for evaluation
- `instances` contains the programs submitted by the students
- `analysis` contains log files from FormHe executions and R code to analyze those executions
- `analysis/data` contains the log files

## Setup

The recommended environment is a based on a fresh conda installation with Python 3.10, along with the packages specified in `pyproject.toml`.

## Execution

The main executable for FormHe is `formhe/asp_integrated.py`. Using `--help` will show all options available.

The file `helper-scripts/evaluate.py` can be used to evaluate all instances and collect statistics.