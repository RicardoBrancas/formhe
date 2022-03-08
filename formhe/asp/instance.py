import re
from pathlib import Path
from typing import IO

import clingo.ast
from ordered_set import OrderedSet
from tqdm import tqdm
from clingo import Control

from asp import utils
from exceptions.parser_exceptions import InstanceParseException
from exceptions.solver_exceptions import NoGroundTruthException


class Instance:

    def __init__(self, filename: str):
        with open(filename) as f:
            self.contents = f.read()

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
            self.instrumented_ast.append(self.instrumenter.visit(ast))
            self.constantCollector.visit(ast)
            if utils.is_fact(ast):
                self.facts.append(ast)
            elif utils.is_rule(ast):
                self.rules.append(ast)
            elif utils.is_integrity_constraint(ast):
                self.integrity_constraints.append(ast)
            else:
                self.others.append(ast)

        clingo.ast.parse_string(self.contents, ast_callback)

        try:
            self.ground_truth_file = Path(filename).resolve().parent / re.search('%formhe-groundtruth:(.*)', self.contents)[1]
            self.ground_truth_file = self.ground_truth_file.resolve()
            self.ground_truth = Instance(self.ground_truth_file)
            self.has_gt = True
        except:
            self.has_gt = False

        self.constants = self.constantCollector.constants.items
        self.cores = OrderedSet()
        self.gt_cores = OrderedSet()
        self.answer_sets = OrderedSet()

    def get_control(self, max_sols=0):
        ctl = Control([f'{max_sols}'])
        ctl.add('base', [], self.contents)
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

    def print_answer_sets(self):
        for a in sorted(self.answer_sets, key=len):
            print(a)

    def print_cores(self):
        for c in sorted(self.cores, key=len):
            print(c)

    def print_gt_cores(self):
        for c in sorted(self.gt_cores, key=len):
            print(c)
