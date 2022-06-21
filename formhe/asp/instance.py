import re
from pathlib import Path

import clingo.ast
from clingo import Control
from ordered_set import OrderedSet
from tqdm import tqdm

from formhe.asp import utils
from formhe.exceptions.solver_exceptions import NoGroundTruthException


class Instance:

    def __init__(self, filename: str = None, ast: clingo.ast.AST = None, skips: list[int] = None):
        if (filename is not None and ast is not None) or (filename is None and ast is None):
            raise ValueError("Need to supply exactly one of {filename, ast}.")

        if filename is not None:
            with open(filename) as f:
                self.contents = f.read()
        elif ast is not None:
            self.contents = '\n'.join(map(str, ast))

        self.others = []
        self.facts = []
        self.rules = []
        self.integrity_constraints = []
        self.constantCollector = utils.ConstantCollector()
        self.instrumenter = utils.Instrumenter()
        self.ast = []
        self.instrumented_ast = []

        def ast_callback(ast):
            self.ast.append(ast)
            self.constantCollector.visit(ast)
            if utils.is_fact(ast):
                self.facts.append(ast)
            elif utils.is_rule(ast):
                self.rules.append(ast)
            elif utils.is_integrity_constraint(ast):
                self.integrity_constraints.append(ast)
            else:
                self.others.append(ast)

        def ast_instrumenter(ast):
            self.instrumented_ast.append(self.instrumenter.visit(ast))

        clingo.ast.parse_string(self.contents, ast_callback)
        clingo.ast.parse_string(self.contents, ast_instrumenter)

        if skips:
            self.ast = [ast for i, ast in enumerate(self.ast) if i not in skips]
            self.instrumented_ast = [ast for i, ast in enumerate(self.instrumented_ast) if i not in skips]

        try:
            self.ground_truth_file = Path(filename).resolve().parent / re.search('%formhe-groundtruth:(.*)', self.contents)[1]
            self.ground_truth_file = self.ground_truth_file.resolve()
            self.ground_truth = Instance(self.ground_truth_file)
            self.has_gt = True
        except:
            self.has_gt = False

        try:
            self.mcs_query = re.search('%formhe-mcs-query:(.*)', self.contents)[1]
        except:
            pass

        self.constants = self.constantCollector.constants.items
        self.cores = OrderedSet()
        self.gt_cores = OrderedSet()
        self.answer_sets = OrderedSet()

    def get_control(self, max_sols=0, *args):
        ctl = Control([f'{max_sols}'])
        ctl.add('base', [], self.contents)
        for arg in args:
            ctl.add('base', [], arg)
        ctl.ground([('base', [])])
        return ctl

    def get_instrumented_control(self, max_sols=0, *args):
        ctl = Control([f'{max_sols}'])
        ctl.add('base', [], '\n'.join(map(str, self.instrumented_ast)))
        for arg in args:
            ctl.add('base', [], arg)
        ctl.ground([('base', [])])
        return ctl

    def find_wrong_models(self, max_sols):
        if not self.has_gt:
            raise NoGroundTruthException()

        ctl = self.get_control(max_sols=max_sols)

        t = tqdm(total=max_sols, desc='Computing models')

        with ctl.solve(yield_=True) as handle:
            for m in handle:
                t.update()

                symbols = list(m.symbols(atoms=True))

                gt_ctl = self.ground_truth.get_control(1)

                cores = []
                gt_ctl.solve([(symbol, True) for symbol in symbols], on_core=lambda c: cores.append(c), on_model=lambda m: self.answer_sets.add(m.symbols(atoms=True)))

                core = []

                if len(cores) == 0:
                    continue

                for sym in gt_ctl.symbolic_atoms:
                    if sym.literal in cores[-1]:
                        core.append(sym.symbol)

                self.cores.add(tuple(sorted(core)))

        t.close()

        ctl = self.ground_truth.get_control(max_sols=max_sols)

        t = tqdm(total=max_sols, desc='Computing gt models')

        with ctl.solve(yield_=True) as handle:
            for m in handle:
                t.update()

                symbols = list(m.symbols(atoms=True))

                gt_ctl = self.get_control(1)

                cores = []
                gt_ctl.solve([(symbol, True) for symbol in symbols], on_core=lambda c: cores.append(c), on_model=lambda m: self.answer_sets.add(m.symbols(atoms=True)))

                core = []

                if len(cores) == 0:
                    continue

                for sym in gt_ctl.symbolic_atoms:
                    if sym.literal in cores[-1]:
                        core.append(sym.symbol)

                self.gt_cores.add(tuple(sorted(core)))

        t.close()

    def generate_correct_models(self, max_sols):
        if not self.has_gt:
            raise NoGroundTruthException()

        ctl = self.ground_truth.get_control(max_sols=max_sols)

        t = tqdm(total=max_sols, desc='Computing models')

        def model_callback(m):
            self.answer_sets.add(m.symbols(atoms=True))
            t.update()

        ctl.solve(on_model=model_callback)

        t.close()

    def find_mcs(self, query, minimum=False):
        instrumenter_vars = self.instrumenter.instrumenter_vars
        n_vars = len(instrumenter_vars)

        generate_vars = '0 { ' + '; '.join(map(str, instrumenter_vars)) + ' } ' + str(n_vars) + '.'

        print()
        print(generate_vars)
        print()

        clause_s_set = OrderedSet()
        clause_s = ''
        clause_d = ''
        clause_r = ''

        while True:
            if not minimum:
                ctl = self.get_instrumented_control(1, generate_vars, clause_s, clause_d, query)
            else:
                ctl = self.get_instrumented_control(1, generate_vars, clause_r, query)

            unsat = True

            with ctl.solve(yield_=True) as handle:
                for m in handle:
                    unsat = False
                    unsatisfied = []

                    for var in instrumenter_vars:
                        if not m.contains(var):
                            unsatisfied.append(var)
                        else:
                            clause_s_set.add(var)

                    clause_s = ' '.join(map(lambda x: f'{x}.', clause_s_set))
                    clause_d = '1 { ' + '; '.join(map(str, unsatisfied)) + ' } ' + str(len(unsatisfied)) + '.'

                    clause_r = str(len(clause_s_set) + 1) + ' { ' + '; '.join(map(str, instrumenter_vars)) + ' } ' + str(n_vars) + '.'

                    if not minimum:
                        print(clause_s)
                        print(clause_d)
                    else:
                        print(clause_r)
                    print()

            if unsat:
                return [self.instrumenter.instrumenter_var_map[var] for var in unsatisfied]

    def print_answer_sets(self):
        for a in sorted(self.answer_sets, key=len):
            print(a)

    def print_cores(self):
        for c in sorted(self.cores, key=len):
            print(c)

    def print_gt_cores(self):
        for c in sorted(self.gt_cores, key=len):
            print(c)
