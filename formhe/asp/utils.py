import itertools

import clingo
import clingo.ast
from ordered_set import OrderedSet


def is_fact(ast: clingo.ast.AST):
    return ast.ast_type == clingo.ast.ASTType.Rule and len(ast.body) == 0


def is_rule(ast: clingo.ast.AST):
    return ast.ast_type == clingo.ast.ASTType.Rule and len(ast.body) != 0 and (
            ast.head.ast_type != clingo.ast.ASTType.Literal or ast.head.atom != clingo.ast.BooleanConstant(0))


def is_integrity_constraint(ast: clingo.ast.AST):
    return ast.ast_type == clingo.ast.ASTType.Rule and len(
        ast.body) != 0 and ast.head.ast_type == clingo.ast.ASTType.Literal and ast.head.atom == clingo.ast.BooleanConstant(
        0)


class Visitor(clingo.ast.Transformer):

    def __init__(self, skips: list[int] = None):
        self.constants = OrderedSet()
        self.predicates = {}
        self.predicates_typed = {}
        self.rule_counter = 0
        self.skips = skips
        self.skipped = []

    def visit_Function(self, function):
        arg_types = []
        add = True
        if function.name:
            self.predicates[function.name] = len(function.arguments)
        for arg in function.arguments:
            if arg.ast_type == clingo.ast.ASTType.SymbolicTerm:
                arg_types.append(arg.symbol.type)
            else:
                add = False

            if arg.ast_type == clingo.ast.ASTType.SymbolicTerm and arg.symbol.type == clingo.SymbolType.Function and len(
                    arg.symbol.arguments) == 0:
                self.constants.append(arg.symbol.name)
        if add:
            self.predicates_typed[function.name] = arg_types
        return function

    def visit_Rule(self, rule):
        self.rule_counter += 1
        try:
            if rule.head.atom.symbol.name == "formhe_definition_begin" or rule.head.atom.symbol.name == "formhe_definition_end":
                return None
        except:
            pass

        if self.skips is not None and self.rule_counter - 1 in self.skips:
            self.skipped.append(rule)
            return None
        else:
            rule.update(**self.visit_children(rule))
            return rule


class Instrumenter(clingo.ast.Transformer):

    def __init__(self):
        self.counter = 0
        self.rule_counter = 0
        self.relaxation_functions = []
        self.relaxations_function_map = {}
        self.disabled = False

    def visit_Rule(self, rule):
        self.rule_counter += 1
        try:
            if rule.head.atom.symbol.name == "formhe_definition_begin":
                self.disabled = True
                return None
        except:
            pass
        try:
            if rule.head.atom.symbol.name == "formhe_definition_end":
                self.disabled = False
                return None
        except:
            pass

        if not self.disabled:
            relaxation_function = clingo.Function(f"_instrumenter", [clingo.Number(self.counter)])
            self.relaxation_functions.append(relaxation_function)
            self.relaxations_function_map[relaxation_function] = self.rule_counter - 1
            rule.body.append(clingo.ast.Literal(rule.location,
                                                clingo.ast.Sign.NoSign,
                                                clingo.ast.SymbolicAtom(
                                                    clingo.ast.Function(rule.location,
                                                                        f"_instrumenter",
                                                                        [clingo.ast.SymbolicTerm(rule.location, clingo.Number(self.counter))],
                                                                        0))))
            self.counter += 1
            return rule
        else:
            return rule

    def assumption_combos(self):
        for combo in itertools.product([True, False], repeat=len(self.relaxation_functions)):
            str = ''
            disabled = []
            for i, (var, val) in enumerate(zip(self.relaxation_functions, combo)):
                if val:
                    str += f'{var}. '
                else:
                    str += f'not {var}. '
                    disabled.append(i)
            yield str, disabled
