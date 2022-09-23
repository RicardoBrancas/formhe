import logging
from functools import cached_property
from typing import TypeVar

import z3

from formhe.trinity.DSL import TyrellSpec
from formhe.trinity.smt_enumerator import SmtEnumerator
from formhe.utils import config

logger = logging.getLogger('formhe.asp.enumerator')

ExprType = TypeVar('ExprType')


class Z3Enumerator(SmtEnumerator):

    def __init__(self, spec: TyrellSpec, depth=None, n_roots=1, predicates_names=None, cores=None, free_vars=None, preset_atoms=None):
        super().__init__(spec, depth, n_roots, predicates_names, cores, free_vars, preset_atoms)

    def init_solver(self):
        solver = z3.SolverFor('QF_NIA')

        solver.set('random_seed', config.get().seed)
        solver.set('unsat_core', True)
        solver.set('core.minimize', True)

        return solver

    @cached_property
    def smt_namespace(self) -> z3:
        original_And = z3.And

        def And(*args):
            if len(args) == 0:
                return True
            if isinstance(args[0], list):
                return And(*args[0])
            if len(args) == 1:
                return args[0]
            else:
                return original_And(*args)

        z3.And = And

        return z3
