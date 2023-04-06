import logging
from typing import Optional, Callable, Sequence, NamedTuple, Any
import itertools

import runhelper
from ordered_set import OrderedSet

from formhe.trinity.DSL import ApplyNode, AtomNode
from formhe.trinity.z3_enumerator import Z3Enumerator
from formhe.utils import iterutils, perf, config
from trinity.smt_enumerator import PresetStatement

logger = logging.getLogger('formhe.asp.enumerator')


class StatementEnumerator:

    def __init__(self, atom_enumerator_constructor: Callable[[Sequence[PresetStatement]], Z3Enumerator], preset_statements: list[PresetStatement], n_atoms=1, free_vars=None, depth=1):
        self.atom_enumerator_constructor = atom_enumerator_constructor
        self.depth = depth

        if n_atoms != 1:
            raise NotImplementedError()

        if free_vars is None:
            free_vars = []
        self.free_vars = OrderedSet(free_vars)

        self.preset_statements = preset_statements

        stmt_combos = []
        for head, body in preset_statements:
            if head is None:
                stmt_combos.append([PresetStatement(False, None, body_combo) for body_combo in iterutils.toggleset_add_one_to_base(body)])
            else:
                if head != 'empty':
                    stmt_combos.append([PresetStatement(True, head_combo, body_combo) for body_combo in iterutils.toggleset_add_one_to_base(body) for head_combo in [head, None]])
                else:
                    stmt_combos.append([PresetStatement(True, None, body_combo) for body_combo in iterutils.toggleset_add_one_to_base(body)])

        self.preset_statement_combinations = itertools.product(*stmt_combos)
        self.next_enum()

    def next_enum(self):
        self.current_preset_statements = next(self.preset_statement_combinations)
        logger.info('Starting enum for %r with depth %d', self.current_preset_statements, self.depth)
        runhelper.timer_start('smt.enum.construction.time')
        self.enumerator = self.atom_enumerator_constructor(self.current_preset_statements)
        runhelper.timer_stop('smt.enum.construction.time')
        self.enumerator.create_no_unsafe_vars_constraints([self.free_vars - self.get_safe_vars(stmt.body) for stmt in self.current_preset_statements])
        self.enumerator.create_force_var_usage_constraints([self.get_all_vars(stmt.head).union(self.get_used_vars(stmt.body)) - self.get_safe_vars(stmt.body)
                                                            for stmt in self.current_preset_statements])

    def get_all_vars(self, expr):
        if isinstance(expr, list) or isinstance(expr, tuple):
            used_vars = OrderedSet()
            for elem in expr:
                used_vars.update(self.get_safe_vars(elem))
            return used_vars
        elif isinstance(expr, ApplyNode):
            used_vars = OrderedSet()
            for elem in expr.args:
                used_vars.update(self.get_all_vars(elem))
            return used_vars
        elif isinstance(expr, AtomNode):
            if expr.production._get_rhs() in self.free_vars:
                return OrderedSet([expr.production._get_rhs()])
            else:
                return OrderedSet()
        elif expr is None:
            return OrderedSet()
        else:
            raise NotImplementedError(type(expr))

    def get_safe_vars(self, expr, parent_is_predicate=False):
        if isinstance(expr, list) or isinstance(expr, tuple):
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
                return OrderedSet([expr.production._get_rhs()])
            else:
                return OrderedSet()
        elif expr is None:
            return OrderedSet()
        else:
            raise NotImplementedError(type(expr))

    def get_used_vars(self, expr):
        if isinstance(expr, list) or isinstance(expr, tuple):
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
                return OrderedSet([expr.production._get_rhs()])
            else:
                return OrderedSet()
        elif expr is None:
            return OrderedSet()
        else:
            raise NotImplementedError(type(expr))

    def __next__(self):
        atom = self.enumerator.next()

        if atom:
            runhelper.timer_start('block.time')
            self.enumerator.update()
            runhelper.timer_stop('block.time')
            return atom

        else:
            try:
                self.next_enum()
            except StopIteration:
                return None
            return next(self)
