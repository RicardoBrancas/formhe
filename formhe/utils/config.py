import argparse
import ast
import base64
import dataclasses
import fcntl
import logging
import os
import re
import sys
import time
import typing
from dataclasses import dataclass, field
from typing import Any, Tuple

from argparse_dataclass import ArgumentParser


def parse_selfeval_lines(s: str) -> Tuple[Tuple[int, ...], ...]:
    if s.strip() == '':
        return ((),)
    if ',' in s:
        elems_s = re.findall(r'\((.*?)\)', s)
        return tuple(ast.literal_eval('(' + e + ',)') if len(e) > 0 else tuple() for e in elems_s)
    else:
        return (tuple(map(int, s.split())),)


def parse_selfeval_fix(s: str) -> Tuple[str, ...]:
    if '"' in s:
        return ast.literal_eval('(' + s + ')')
    else:
        return (s,)


@dataclass(frozen=True)
class Config:
    # inputs
    input_file: str = field(metadata={'args': ['input_file']})
    problem: str = None
    problems_folder: str = "./problems"
    timestamp: int = field(default_factory=lambda: int(time.time()), metadata=dict(type=int))

    # core
    seed: Any = 42
    logging_level: str = 'DEBUG'
    eval_params: str = field(metadata=dict(required=False, type=lambda x: base64.b64decode(x.encode('utf-8')).decode('utf-8')), default='')
    no_enumerator_debug: bool = False
    no_stdin_instance: bool = False
    exit_after_fault_localization: bool = False
    drop_stderr: bool = False
    print_only_first_test_case: bool = False
    external_fl: list[int] = field(metadata=dict(nargs='*', type=int), default=None)
    model_feedback_n: int = 5
    ignore_timestamps: bool = False
    hints_only: bool = False
    fl_watch_file: str = None

    # llm
    llm_url: str = "http://localhost:3000"
    llm_timeout: int = 300
    fl_prompt_version: int = 3
    repair_prompt_version: int = 3

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
    enum_all_answer_sets: bool = False

    # parameters
    minimum_depth: int = 2
    maximum_depth: int = 5
    dynamic_depth: bool = True
    minimum_mutations: int = 1
    maximum_mutations: int = 5
    empty_statements: int = 2
    additional_body_nodes: int = 1
    bandit_starting_epsilon: float = 1
    bandit_epsilon_multiplier: float = 0.9999
    bandit_exploration_count: int = 5000
    extra_vars: int = 2
    skip_mcs_negative_non_relaxed: bool = False
    skip_mcs_negative_relaxed: bool = False
    use_mcs_positive: bool = False
    skip_mcs_positive_conditional: bool = False
    use_mfl: bool = False
    skip_mcs_line_pairings: bool = False
    enable_arithmetic: bool = False
    disable_classical_negation: bool = False
    use_sbfl: bool = False
    mcs_sorting_method: str = field(metadata=dict(choices=['hit-count', 'hit-count-avg', 'hit-count-normalized', 'hit-count-smallest', 'random', 'random-smallest', 'none', 'none-smallest']), default='hit-count')
    disable_mutation_node_expansion: bool = False
    skip_llm_fl: bool = False
    skip_llm_repair: bool = False
    mutate_llm_attempt: bool = False
    # mcs_timeout: int = 10
    iterative_llm: bool = True
    max_llm_iterations: int = 3
    mutation_based_repair: bool = True
    enable_pool_operator: bool = False
    use_canon_reference: bool = False

    # heuristics
    line_matching_threshold: int = 3

    # evaluation
    selfeval_lines: tuple[tuple[int]] = field(metadata=dict(type=parse_selfeval_lines), default=None)
    selfeval_deleted_lines: int = None
    selfeval_changes_generate: list[int] = field(metadata=dict(nargs='*', type=int), default=None)
    selfeval_changes_generate_n: int = None
    selfeval_changes_generate_n_unique: int = None
    selfeval_changes_test: list[int] = field(metadata=dict(nargs='*', type=int), default=None)
    selfeval_changes_test_n: int = None
    selfeval_changes_test_n_unique: int = None
    simulate_fault_localization: bool = False
    selfeval_fix_test: bool = False
    selfeval_fix: tuple[str] = field(metadata=dict(type=parse_selfeval_fix), default=None)
    test_partial_fault: bool = False

    def process_overrides(self, overrides, suppress_override_message):
        for key, value in overrides.items():
            if not suppress_override_message:
                logger.info('Overriding local config %s: %s', key, value)

            for field in dataclasses.fields(Config):
                if field.name == key:
                    if field.type is list or typing.get_origin(field.type) is list:
                        if field.metadata and 'type' in field.metadata and callable(field.metadata['type']):
                            object.__setattr__(self, key, [field.metadata['type'](v) for v in value.split(' ') if v])
                        else:
                            object.__setattr__(self, key, [v for v in value.split(' ') if v])
                    elif field.type is bool:
                        try:
                            object.__setattr__(self, key, strtobool(value))
                        except:
                            object.__setattr__(self, key, value)
                    elif field.metadata and 'type' in field.metadata and callable(field.metadata['type']):
                        try:
                            object.__setattr__(self, key, field.metadata['type'](value))
                        except Exception as e:
                            object.__setattr__(self, key, value)
                    else:
                        object.__setattr__(self, key, value)

                    break
            else:
                raise AttributeError(key)


parser = ArgumentParser(Config, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
res = parser.parse_known_args()
_config: Config = res[0]

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
