import clingo
import clingo.ast
from ordered_set import OrderedSet


def is_fact(ast: clingo.ast.AST):
    return ast.ast_type == clingo.ast.ASTType.Rule and len(ast.body) == 0


def is_rule(ast: clingo.ast.AST):
    return ast.ast_type == clingo.ast.ASTType.Rule and len(ast.body) != 0 and (ast.head.ast_type != clingo.ast.ASTType.Literal or ast.head.atom != clingo.ast.BooleanConstant(0))


def is_integrity_constraint(ast: clingo.ast.AST):
    return ast.ast_type == clingo.ast.ASTType.Rule and len(ast.body) != 0 and ast.head.ast_type == clingo.ast.ASTType.Literal and ast.head.atom == clingo.ast.BooleanConstant(0)


class ConstantCollector(clingo.ast.Transformer):

    def __init__(self):
        self.constants = OrderedSet()

    def visit_Function(self, function):
        for arg in function.arguments:
            if arg.ast_type == clingo.ast.ASTType.SymbolicTerm and arg.symbol.type == clingo.SymbolType.Function and len(arg.symbol.arguments) == 0:
                self.constants.append(arg.symbol.name)
        return function


class Instrumenter(clingo.ast.Transformer):

    def __init__(self):
        pass

    def visit_Rule(self, rule):
        print(rule)
        return rule
