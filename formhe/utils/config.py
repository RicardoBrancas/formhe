import argparse
import base64
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from argparse_dataclass import ArgumentParser


@dataclass(frozen=True)
class Config:
    # inputs
    input_file: str = field(metadata={'args': ['input_file']})
    groundtruth: str = field(metadata={'args': ['--ground-truth']}, default=None)
    mcs_query: str = field(metadata={'args': ['--query']}, default=None)
    domain_predicates: list = field(metadata=dict(nargs='+', type=str), default=None)

    # core
    seed: Any = 42
    logging_level: str = 'DEBUG'
    eval_params: str = field(metadata=dict(required=False, type=lambda x: base64.b64decode(x.encode('utf-8')).decode('utf-8')), default='')
    no_enumerator_debug: bool = False

    # predicates
    disable_commutative_predicate: bool = False

    # optimizations
    allow_constant_expressions: bool = False
    allow_unsafe_vars: bool = False
    no_bind_free_semantic_vars: bool = False
    no_semantic_constraints: bool = False

    # parameters
    minimum_depth: int = 2
    maximum_depth: int = 5
    n_gt_sols_generated: int = 200
    n_gt_sols_checked: int = 5
    n_candidate_sols_checked: int = 5
    model_cache_size: int = 512
    bandit_starting_epsilon: float = 1
    bandit_epsilon_multiplier: float = 0.9999
    bandit_exploration_count: int = 5000
    limit_enumerated_atoms: int = None


parser = ArgumentParser(Config, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_config: Config = parser.parse_args()

logging.basicConfig(level=_config.logging_level, format='%(relativeCreated)8d | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('formhe')

stdin_content = sys.stdin.readlines()


def store(conf: Config):
    global _config
    _config = conf


def get() -> Config:
    global _config
    return _config
