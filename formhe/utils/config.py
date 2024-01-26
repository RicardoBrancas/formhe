import argparse
import base64
import logging
import os
import sys
import fcntl
from dataclasses import dataclass, field
from typing import Any

from argparse_dataclass import ArgumentParser


@dataclass(frozen=True)
class Config:
    # inputs
    input_file: str = field(metadata={'args': ['input_file']})
    groundtruth: str = field(metadata={'args': ['--ground-truth']}, default=None)
    mcs_query: str = field(metadata={'args': ['--query']}, default=None)
    domain_predicates: list = field(metadata=dict(nargs='*', type=str), default=None)
    optimization_problem: bool = False

    # core
    seed: Any = 42
    logging_level: str = 'DEBUG'
    eval_params: str = field(metadata=dict(required=False, type=lambda x: base64.b64decode(x.encode('utf-8')).decode('utf-8')), default='')
    instance_base64: list[str] = field(metadata=dict(required=False, nargs='+', type=lambda s: base64.b64decode(s.encode('utf-8')).decode('utf-8')), default='')
    no_enumerator_debug: bool = False
    no_stdin_instance: bool = False
    exit_after_fault_localization: bool = False
    drop_stderr: bool = False
    print_only_first_test_case: bool = False

    # predicates
    disable_commutative_predicate: bool = False
    disable_distinct_args_predicate: bool = False

    # optimizations
    block_constant_expressions: bool = False
    allow_unsafe_vars: bool = False
    allow_not_generated_predicates: bool = False
    enable_redundant_arithmetic_ops: bool = False
    disable_head_empty_or_non_constant_constraint: bool = False
    disable_no_dont_care_in_head_constraint: bool = False

    # parameters
    minimum_depth: int = 2
    maximum_depth: int = 5
    bandit_starting_epsilon: float = 1
    bandit_epsilon_multiplier: float = 0.9999
    bandit_exploration_count: int = 5000
    extra_vars: int = 2
    skip_mcs_negative_non_relaxed: bool = False
    skip_mcs_negative_relaxed: bool = False
    use_mcs_strong_negative_non_relaxed: bool = False # TODO fix me misleading names
    use_mcs_strong_negative_relaxed: bool = False
    use_mcs_positive: bool = False
    use_mfl: bool = False
    skip_mcs_line_pairings: bool = False
    enable_arithmetic: bool = False
    disable_classical_negation: bool = False
    use_sbfl: bool = False
    mcs_sorting_method: str = field(metadata=dict(choices=['hit-count', 'hit-count-normalized', 'hit-count-smallest', 'random', 'random-smallest', 'none', 'none-smallest']), default='none-smallest')

    # heuristics

    # evaluation
    selfeval_lines: list[int] = field(metadata=dict(nargs='*', type=int), default=None)
    selfeval_deleted_lines: int = None
    selfeval_changes_generate: list[int] = field(metadata=dict(nargs='*', type=int), default=None)
    selfeval_changes_generate_n: int = None
    selfeval_changes_generate_n_unique: int = None
    selfeval_changes_test: list[int] = field(metadata=dict(nargs='*', type=int), default=None)
    selfeval_changes_test_n: int = None
    selfeval_changes_test_n_unique: int = None
    simulate_fault_localization: bool = False
    selfeval_fix_test: bool = False
    selfeval_fix: str = None


parser = ArgumentParser(Config, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_config: Config = parser.parse_args()

if _config.drop_stderr:
    f = open(os.devnull, 'w')
    sys.stderr = f

logging.basicConfig(level=_config.logging_level, format='%(relativeCreated)8d | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('formhe')

if not _config.no_stdin_instance:
    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    try:
        lines = sys.stdin.readlines()
        lines.insert(0, 'formhe_definition_begin.')
        lines.append('formhe_definition_end.')
        stdin_content = lines
    except:
        stdin_content = []
else:
    stdin_content = []


def store(conf: Config):
    global _config
    _config = conf


def get() -> Config:
    global _config
    return _config


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))
