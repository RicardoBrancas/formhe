import io
import sys
from collections import defaultdict
from typing import Union, TextIO

import clingo.ast
from clingo.ast import *

from utils.graph import Graph


class HighlighterPrinter:

    def __init__(self, stream: TextIO, resolution: int):
        self.stream = stream
        self.division_multiplier = 0
        self.resolution = resolution

    def start(self, n_divisions: float):
        if n_divisions != 0:
            self.division_multiplier = self.resolution / n_divisions

    def start_color(self, weight: float):
        pass

    def process_character(self, char: str):
        print(char, end='', file=self.stream)

    def end_color(self):
        pass

    def end(self):
        print()


class TTYPrinter(HighlighterPrinter):

    def __init__(self, stream: TextIO):
        super().__init__(stream, 255)
        self.color_stacks = ['\x1b[0m']

    def start_color(self, weight: float):
        color = f'\x1b[38;2;0;{int(weight * self.division_multiplier)};0m'
        print(color, end='', file=self.stream)
        self.color_stacks.append(color)

    def end_color(self):
        self.color_stacks.pop()
        print(self.color_stacks[-1], end='', file=self.stream)

    def end(self):
        print('\x1b[0m', file=self.stream)


class MarkdownPrinter(HighlighterPrinter):

    def __init__(self, stream: TextIO):
        super().__init__(stream, 255)
        self.open_spans = 0

    def start(self, n_divisions: float):
        super().start(n_divisions)
        print('<pre>', end='', file=self.stream)

    def start_color(self, weight: float):
        self.open_spans += 1
        print(f'<span style="color: #00{int(weight * self.division_multiplier):02x}00">', end='', file=self.stream)

    def end_color(self):
        self.open_spans -= 1
        print('</span>', end='', file=self.stream)

    def end(self):
        while self.open_spans > 0:
            print('</span>', end='', file=self.stream)
            self.open_spans -= 1
        print('</pre>', file=self.stream)


class AspHighlithingVisitor:

    def __init__(self):
        self.out = io.StringIO()
        self.rule_counter = 0
        self.rule_map = dict()
        self.graph = Graph()
        self.location_map = defaultdict(list)

    def __del__(self):
        print(self.out.getvalue())

    def highlight(self, path: str, cores):
        clingo.ast.parse_files([path], lambda x: self.visit(x, cores))
        print(self.graph.edges)
        for core in cores:
            for atom in core:
                self.graph.traverse_graph_update_weights(f'{atom.name}/{len(atom.arguments)}', 1)
        print(self.graph.weights)
        self.print_colored(path)

    def print_colored(self, path):
        col, row = 1, 1
        printer = TTYPrinter(sys.stdout)
        printer.start(max(self.graph.weights.values()) if len(self.graph.weights) != 0 else 0)
        with open(path) as file:
            for char in file.read():
                for node, weight in self.graph.weights.items():
                    for location in self.location_map[node]:
                        if location.end.line == row and location.end.column == col:
                            printer.end_color()

                        if location.begin.line == row and location.begin.column == col:
                            printer.start_color(weight)

                printer.process_character(char)

                if char == '\n':
                    row += 1
                    col = 1
                else:
                    col += 1

        printer.end()

    def visit(self, ast: Union[list, ASTSequence, AST], cores):
        if isinstance(ast, list):
            return [self.visit(sub_ast, cores) for sub_ast in ast]

        elif isinstance(ast, ASTSequence):
            [self.visit(sub_ast, cores) for sub_ast in ast]
            return ast

        elif ast is None:
            return ast

        else:
            method_name = 'visit_' + str(ast.ast_type).replace('ASTType.', '')
            if hasattr(self, method_name):
                return getattr(self, method_name)(ast, cores)
            else:
                raise NotImplementedError(str(ast.ast_type).replace('ASTType.', ''))

    def visit_Program(self, ast: AST, cores):
        self.visit(ast.parameters, cores)
        return ast

    def visit_Rule(self, ast: AST, cores):
        self.current_rule = ast
        self.rule_map[ast] = self.rule_counter
        self.rule_counter += 1
        self.visit(ast.head, cores)
        for body_ast in ast.body:
            self.visit(body_ast, cores)
        self.current_rule = None
        return ast

    def visit_Literal(self, ast: AST, cores):
        self.visit(ast.atom, cores)
        return ast

    def visit_SymbolicAtom(self, ast: AST, cores):
        self.visit(ast.symbol, cores)
        return ast

    def visit_Definition(self, ast: AST, cores):
        self.visit(ast.value, cores)
        return ast

    def visit_SymbolicTerm(self, ast: AST, cores):
        print(ast)
        return ast

    def visit_Aggregate(self, ast: AST, cores):
        self.visit(ast.left_guard, cores)
        for elem in ast.elements:
            self.visit(elem, cores)
        self.visit(ast.right_guard, cores)
        return ast

    def visit_Guard(self, ast: AST, cores):
        self.visit(ast.term, cores)
        return ast

    def visit_ConditionalLiteral(self, ast: AST, cores):
        self.visit(ast.literal, cores)
        self.visit(ast.condition, cores)
        return ast

    def visit_Function(self, ast: AST, cores):
        self.current_function = ast
        self.location_map[f'{ast.name}/{len(ast.arguments)}'].append(ast.location)
        for arg in ast.arguments:
            self.visit(arg, cores)
        self.current_function = None
        return ast

    def visit_Variable(self, ast: AST, cores):
        self.location_map[(self.rule_map[self.current_rule], ast.name)].append(ast.location)
        if self.current_function:
            self.graph.add_var(f'{self.current_function.name}/{len(self.current_function.arguments)}', (self.rule_map[self.current_rule], ast.name), .8)
        else:
            pass
        return ast

    def visit_BooleanConstant(self, ast: AST, cores):
        return ast

    def visit_Comparison(self, ast: AST, cores):
        self.visit(ast.term, cores)
        for guard in ast.guards:
            self.visit(guard, cores)
        return ast

    def visit_ShowSignature(self, ast: AST, cores):
        return ast

    def visit_Interval(self, ast: AST, cores):
        self.visit(ast.left, cores)
        self.visit(ast.right, cores)
        return ast

    def visit_Pool(self, ast: AST, cores):
        for arg in ast.arguments:
            self.visit(arg, cores)
        return ast


class RuleScoreCalculator(clingo.ast.Transformer):

    def __init__(self, graph_weights, scorer: 'Scorer'):
        self.rule_counter = 0
        self.rule_map = dict()
        self.scorer = scorer
        self.graph_weights = graph_weights

    def visit_Rule(self, ast):
        self.current_rule = ast
        self.rule_map[ast] = self.rule_counter
        self.scorer.update_score(self.rule_map[ast])
        self.rule_counter += 1
        self.visit_children(ast)
        return ast

    def visit_Function(self, ast):
        fun_id = f'{ast.name}/{len(ast.arguments)}'
        if fun_id in self.graph_weights:
            self.scorer.update_score(self.rule_map[self.current_rule], self.graph_weights[fun_id])
        self.visit_children(ast)
        return ast

    def visit_Variable(self, ast):
        var_id = (self.rule_map[self.current_rule], ast.name)
        if var_id in self.graph_weights:
            self.scorer.update_score(self.rule_map[self.current_rule], self.graph_weights[var_id])
        return ast


class Scorer:

    def update_score(self, rule_id, score_update=None):
        raise NotImplementedError()

    def get_scores(self):
        raise NotImplementedError()


class GeometricScorer(Scorer):

    def __init__(self):
        self.rule_scores = defaultdict(lambda: 1)
        self.rule_updates = defaultdict(lambda: 0)

    def update_score(self, rule_id, score_update=None):
        if score_update is None:
            score_update = 1
        self.rule_scores[rule_id] *= score_update
        self.rule_updates[rule_id] += 1

    def get_scores(self):
        scores = []
        for rule in self.rule_scores.keys():
            scores.append((rule, self.rule_scores[rule] ** (1 / self.rule_updates[rule])))
        return scores


class ArithmeticScorer(Scorer):

    def __init__(self):
        self.rule_scores = defaultdict(lambda: 0)
        self.rule_updates = defaultdict(lambda: 0)

    def update_score(self, rule_id, score_update=None):
        if score_update is None:
            score_update = 0
        self.rule_scores[rule_id] += score_update
        self.rule_updates[rule_id] += 1

    def get_scores(self):
        scores = []
        for rule in self.rule_scores.keys():
            scores.append((rule, self.rule_scores[rule] / self.rule_updates[rule]))
        return scores


class MaxScorer(Scorer):

    def __init__(self):
        self.rule_scores = defaultdict(lambda: 0)

    def update_score(self, rule_id, score_update=None):
        if score_update is None:
            score_update = 0
        self.rule_scores[rule_id] = max(score_update, self.rule_scores[rule_id])

    def get_scores(self):
        scores = []
        for rule in self.rule_scores.keys():
            scores.append((rule, self.rule_scores[rule]))
        return scores
