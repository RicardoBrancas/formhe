import copy
import dataclasses
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import clingo.ast
from clingo import Control, Symbol
from ordered_set import OrderedSet
from tqdm import tqdm

from formhe.asp.utils import Visitor, Instrumenter, is_fact, is_rule, is_integrity_constraint
from formhe.exceptions.solver_exceptions import NoGroundTruthException
from formhe.utils import perf, config
from formhe.utils.ml import MultiArmedBandit

logger = logging.getLogger('formhe.asp.instance')


class Instance:

    def __init__(self, filename: str = None, ast: clingo.ast.AST = None, skips: list[int] = None, parent_config: config.Config = None):
        if (filename is not None and ast is not None) or (filename is None and ast is None):
            raise ValueError("Need to supply exactly one of {filename, ast}.")

        if filename is not None:
            with open(filename) as f:
                self.contents = f.read() + '\n'.join(config.stdin_content)
        elif ast is not None:
            self.contents = '\n'.join(map(str, ast))

        self.others = []
        self.facts = []
        self.rules = []
        self.integrity_constraints = []
        self.constantCollector = Visitor(skips)
        self.instrumenter = Instrumenter()
        self.ast = []
        self.instrumented_ast = []
        self.var_ctr = 0

        if not parent_config:
            self.config = copy.copy(config.get())
        else:
            self.config = copy.copy(parent_config)

        matches = re.findall(r'%formhe-([a-zA-Z-_]*):(.*)', self.contents)
        for override in matches:
            key = override[0].replace('-', '_')
            value = override[1]
            logger.info('Overriding local config %s: %s', key, value)
            field_found = False
            for field in dataclasses.fields(config.Config):
                if field.name == key:
                    field_found = True
                    break
            if not field_found:
                raise AttributeError(key)
            if field.type is list:
                object.__setattr__(self.config, key, list(value.split(' ')))
            else:
                object.__setattr__(self.config, key, value)

        if self.config.domain_predicates:
            for pred in self.config.domain_predicates:
                self.contents += f'\n#show {pred}.'

        def ast_callback(ast):
            ast = self.constantCollector.visit(ast)
            if ast is not None:
                self.ast.append(ast)
                if is_fact(ast):
                    self.facts.append(ast)
                elif is_rule(ast):
                    self.rules.append(ast)
                elif is_integrity_constraint(ast):
                    self.integrity_constraints.append(ast)
                else:
                    self.others.append(ast)

        def ast_instrumenter(ast):
            instrumented_ast = self.instrumenter.visit(ast)
            if instrumented_ast is not None:
                self.instrumented_ast.append(instrumented_ast)

        clingo.ast.parse_string(self.contents, ast_callback)
        clingo.ast.parse_string(self.contents, ast_instrumenter)

        if self.config.groundtruth:
            self.ground_truth_file = Path(filename).resolve().parent / self.config.groundtruth
            self.ground_truth_file = self.ground_truth_file.resolve()
            config_tmp = copy.copy(self.config)
            object.__setattr__(config_tmp, 'groundtruth', None)
            self.ground_truth = Instance(self.ground_truth_file, parent_config=config_tmp)
            self.has_gt = True
        else:
            self.has_gt = False

        self.constants = self.constantCollector.constants.items
        self.cores = OrderedSet()
        self.gt_cores = OrderedSet()
        self.gt_unsat_models = OrderedSet()
        self.answer_sets = OrderedSet()
        self.answer_sets_asm: MultiArmedBandit[Sequence[Tuple[Symbol, bool]]] = MultiArmedBandit()
        self.control = self.get_control()

    def get_control(self, *args, max_sols=0, instrumented=False, clingo_args: list = None):
        if clingo_args is None:
            clingo_args = []
        ctl = Control(clingo_args + [f'{max_sols}'], logger=lambda x, y: None)
        with clingo.ast.ProgramBuilder(ctl) as bld:
            if not instrumented:
                for stm in self.ast:
                    bld.add(stm)
            else:
                for stm in self.instrumented_ast:
                    bld.add(stm)
            for arg in args:
                clingo.ast.parse_string(arg, bld.add, logger=lambda x, y: None)
        perf.timer_start(perf.GROUNDING_TIME)
        ctl.ground([('base', [])])
        perf.timer_stop(perf.GROUNDING_TIME)
        return ctl

    @lru_cache(config.get().model_cache_size)
    def check_model(self, model: Iterable[Symbol]) -> bool:
        res = self.control.solve(assumptions=[(x, True) for x in model])
        if res.unsatisfiable:
            perf.counter_inc(perf.UNIQUE_UNSAT)
            return False
        elif res.satisfiable:
            perf.counter_inc(perf.UNIQUE_SAT)
            return True
        raise ValueError()

    def find_wrong_models(self, max_sols):
        if not self.has_gt:
            raise NoGroundTruthException()

        ctl = self.get_control(max_sols=max_sols)

        t = tqdm(total=max_sols, desc='Computing models')

        with ctl.solve(yield_=True) as handle:
            for m in handle:
                t.update()

                symbols = list(m.symbols(atoms=True))

                gt_ctl = self.ground_truth.get_control(max_sols=1)

                cores = []
                gt_ctl.solve([(symbol, True) for symbol in symbols], on_core=lambda c: cores.append(c),
                             on_model=lambda m: self.answer_sets.add(m.symbols(atoms=True)))

                core = []

                if len(cores) == 0:
                    continue

                for sym in gt_ctl.symbolic_atoms:
                    if sym.literal in cores[-1]:
                        core.append(sym.symbol)

                self.cores.add(tuple(sorted(core)))

        t.close()

        self.test_gt_answer_sets(max_sols)

    def test_gt_answer_sets(self, max_tests=None, wanted_cores=None, wanted_models=None):
        if max_tests is None and wanted_cores is None and wanted_models is None:
            raise ValueError()

        if wanted_cores is not None or wanted_models is not None:
            max_tests = 0

        ctl = self.ground_truth.get_control(max_sols=max_tests)

        cores_found = 0
        models_found = 0

        t = tqdm(total=max_tests if max_tests != 0 else None, desc='Testing gt answer sets')
        with ctl.solve(yield_=True) as handle:
            for m in handle:
                t.update()

                symbols = list(m.symbols(shown=True))

                gt_ctl = self.get_control(max_sols=1)

                def on_model(m: clingo.Model):
                    nonlocal models_found
                    models_found += 1
                    self.answer_sets.add(m.symbols(atoms=True))

                cores = []
                gt_ctl.solve([(symbol, True) for symbol in symbols], on_core=lambda c: cores.append(c), on_model=on_model)

                if wanted_models is not None and models_found >= wanted_models:
                    return

                core = []
                unsat_model = []

                if len(cores) == 0:
                    continue

                for sym in gt_ctl.symbolic_atoms:
                    if sym.literal in cores[-1]:
                        core.append(sym.symbol)

                for sym in symbols:
                    unsat_model.append(sym)

                self.gt_cores.add(tuple(sorted(core)))
                self.gt_unsat_models.add(tuple(sorted(unsat_model)))

                cores_found += 1
                if wanted_cores is not None and cores_found >= wanted_cores:
                    return

        t.close()

    def generate_correct_models(self, max_sols):
        if not self.has_gt:
            raise NoGroundTruthException()

        ctl = self.ground_truth.get_control(max_sols=max_sols)

        t = tqdm(total=max_sols, desc='Computing models')

        def model_callback(m):
            tmp = m.symbols(shown=True)
            self.answer_sets.add(tuple(sorted([x for x in tmp])))
            self.answer_sets_asm.add_bandit(frozenset((x, True) for x in tmp))
            t.update()

        ctl.solve(on_model=model_callback)

        t.close()

    def find_mcs(self):
        instrumenter_vars = self.instrumenter.relaxation_functions
        n_vars = len(instrumenter_vars)

        generate_vars = '0 { ' + '; '.join(map(str, instrumenter_vars)) + ' } ' + str(n_vars) + '.'

        clause_r = ''

        if self.config.mcs_query:
            negated_query = ' '.join([':- not ' + x + '.' for x in self.config.mcs_query.split('.') if x])
        else:
            self.test_gt_answer_sets(wanted_cores=1)
            if not self.gt_unsat_models:
                return
            logger.info('No MCS query supplied. Using the following unsat model: %s', str(self.gt_unsat_models[0]))
            negated_query = ' '.join([':- not ' + str(x) + '.' for x in self.gt_unsat_models[0] if x])

        logger.info('Transformed query: %s', negated_query)

        logger.info('Starting MCS iterations')
        satisfied_vars = []
        last_clause_r = ''
        while True:
            ctl = self.get_control(generate_vars, clause_r, negated_query, max_sols=1, instrumented=True)

            with ctl.solve(yield_=True) as handle:
                res = handle.get()

                if res.satisfiable:
                    m = handle.model()
                    unsatisfied_vars = []
                    satisfied_vars = []

                    for var in instrumenter_vars:
                        if not m.contains(var):
                            unsatisfied_vars.append(var)
                        else:
                            satisfied_vars.append(var)

                    logger.debug(clause_r)
                    logger.debug(m)
                    logger.debug(' '.join(str(x) for x in satisfied_vars))

                    last_clause_r = f'{len(satisfied_vars)} {{ {"; ".join(map(str, instrumenter_vars))} }} {n_vars}.'
                    clause_r = f'{len(satisfied_vars) + 1} {{ {"; ".join(map(str, instrumenter_vars))} }} {n_vars}.'

                elif res.unsatisfiable:
                    logger.info('MCS loop failed for %d vars', len(satisfied_vars) + 1)
                    logger.info('Iterating MCSs of size %d ', n_vars - len(satisfied_vars))
                    ctl = self.get_control(generate_vars, last_clause_r, negated_query, "#project _instrumenter/1.",
                                           instrumented=True, clingo_args=['--project'])
                    with ctl.solve(yield_=True) as handle:
                        for m in handle:
                            unsatisfied_vars = []
                            for var in instrumenter_vars:
                                if not m.contains(var):
                                    unsatisfied_vars.append(var)
                            logger.debug(' '.join(str(x) for x in unsatisfied_vars))
                            yield [self.instrumenter.relaxations_function_map[var] for var in unsatisfied_vars]
                    return

                else:
                    raise RuntimeError()

    def print_answer_sets(self):
        for a in sorted(self.answer_sets, key=len):
            print(a)

    def print_cores(self):
        for c in sorted(self.cores, key=len):
            print(c)

    def print_gt_cores(self):
        for c in sorted(self.gt_cores, key=len):
            print(c)
