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
        self.rule_counter = 0
        self.skips = skips

    def visit_Function(self, function):
        for arg in function.arguments:
            if arg.ast_type == clingo.ast.ASTType.SymbolicTerm and arg.symbol.type == clingo.SymbolType.Function and len(
                    arg.symbol.arguments) == 0:
                self.constants.append(arg.symbol.name)
        return function

    def visit_Rule(self, rule):
        self.rule_counter += 1
        try:
            if rule.head.atom.symbol.name == "formhe_definition_begin" or rule.head.atom.symbol.name == "formhe_definition_end":
                return None
        except:
            pass

        if self.skips is not None and self.rule_counter - 1 in self.skips:
            return None
        else:
            rule.update(**self.visit_children(rule))
            return rule


class Instrumenter(clingo.ast.Transformer):

    def __init__(self):
        self.counter = 0
        self.rule_counter = 0
        self.instrumenter_vars = []
        self.instrumenter_var_map = {}
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
            instrumenter_var = clingo.ast.Variable(rule.location, f"instrumenter_{self.counter}")
            self.instrumenter_vars.append(clingo.Function(f"instrumenter_{self.counter}"))
            self.instrumenter_var_map[clingo.Function(f"instrumenter_{self.counter}")] = self.rule_counter - 1
            self.counter += 1
            rule.body.append(instrumenter_var)
            return rule
        else:
            return rule

    def assumption_combos(self):
        for combo in itertools.product([True, False], repeat=len(self.instrumenter_vars)):
            str = ''
            disabled = []
            for i, (var, val) in enumerate(zip(self.instrumenter_vars, combo)):
                if val:
                    str += f'{var}. '
                else:
                    str += f'not {var}. '
                    disabled.append(i)
            yield str, disabled
