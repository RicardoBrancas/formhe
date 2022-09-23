import logging
from typing import Optional, Callable, Sequence

from ordered_set import OrderedSet

from formhe.trinity.DSL import ApplyNode, AtomNode
from formhe.trinity.z3_enumerator import Z3Enumerator
from formhe.utils import iterutils, perf, config

logger = logging.getLogger('formhe.asp.enumerator')


class StatementEnumerator:

    def __init__(self, atom_enumerator_constructor: Callable[[int, Sequence], Z3Enumerator], preset_atoms: Optional[list] = None, n_atoms=1, free_vars=None, depth=1):
        self.atom_enumerator_constructor = atom_enumerator_constructor
        self.depth = depth

        if n_atoms != 1:
            raise NotImplementedError()

        if free_vars is None:
            free_vars = []
        self.free_vars = OrderedSet(free_vars)

        if preset_atoms:
            self.preset_atoms = preset_atoms
        else:
            self.preset_atoms = []

        self.preset_atoms_combinations = map(list, iterutils.powerset(self.preset_atoms))
        self.next_enum()

    def next_enum(self):
        self.current_preset_atoms = next(self.preset_atoms_combinations)
        self.enumerator_n_atoms = len(self.preset_atoms) - len(self.current_preset_atoms) if (len(self.preset_atoms) - len(self.current_preset_atoms)) != 0 else 1
        if config.get().limit_enumerated_atoms is not None:
            self.enumerator_n_atoms = min(self.enumerator_n_atoms, config.get().limit_enumerated_atoms)
        logger.info('Starting enum for %d statement(s), %d preset atom(s), %d atom(s), and depth %d', 1, len(self.current_preset_atoms), self.enumerator_n_atoms, self.depth)
        perf.timer_start(perf.SMT_ENUM_CONSTRUCTION_TIME)
        self.enumerator = self.atom_enumerator_constructor(self.enumerator_n_atoms, self.current_preset_atoms)
        perf.timer_stop(perf.SMT_ENUM_CONSTRUCTION_TIME)
        self.enumerator.create_no_unsafe_vars_constraints(self.free_vars - self.get_safe_vars(self.current_preset_atoms))
        self.enumerator.create_force_var_usage_constraints(self.get_used_vars(self.current_preset_atoms) - self.get_safe_vars(self.current_preset_atoms))

    def get_safe_vars(self, expr, parent_is_predicate=False):
        if isinstance(expr, list):
            used_vars = OrderedSet()
            for elem in expr:
                used_vars.update(self.get_safe_vars(elem))
            return used_vars
        elif isinstance(expr, ApplyNode):
            used_vars = OrderedSet()
            for elem in expr.args:
                used_vars.update(self.get_safe_vars(elem, expr.name in self.enumerator.predicate_names))
            return used_vars
        elif isinstance(expr, AtomNode):
            if expr.production._get_rhs() in self.free_vars and parent_is_predicate:
                return [expr.production._get_rhs()]
            else:
                return []
        else:
            raise NotImplementedError(type(expr))

    def get_used_vars(self, expr):
        if isinstance(expr, list):
            used_vars = OrderedSet()
            for elem in expr:
                used_vars.update(self.get_used_vars(elem))
            return used_vars
        elif isinstance(expr, ApplyNode):
            used_vars = OrderedSet()
            for elem in expr.args:
                used_vars.update(self.get_used_vars(elem))
            return used_vars
        elif isinstance(expr, AtomNode):
            if expr.production._get_rhs() in self.free_vars:
                return [expr.production._get_rhs()]
            else:
                return []
        else:
            raise NotImplementedError(type(expr))

    def __next__(self):
        atom = self.enumerator.next()

        if atom:
            perf.timer_start(perf.BLOCK_TIME)
            self.enumerator.update()
            perf.timer_stop(perf.BLOCK_TIME)
            return ApplyNode(self.enumerator.and_production, self.current_preset_atoms + [atom])

        else:
            try:
                self.next_enum()
            except StopIteration:
                return None
            return next(self)
