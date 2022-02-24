import clingo
import clingo.ast


def is_fact(ast: clingo.ast.AST):
    return ast.ast_type == clingo.ast.ASTType.Rule and len(ast.body) == 0


def is_rule(ast: clingo.ast.AST):
    return ast.ast_type == clingo.ast.ASTType.Rule and len(ast.body) != 0 and (ast.head.ast_type != clingo.ast.ASTType.Literal or ast.head.atom != clingo.ast.BooleanConstant(0))


def is_integrity_constraint(ast: clingo.ast.AST):
    return ast.ast_type == clingo.ast.ASTType.Rule and len(ast.body) != 0 and ast.head.ast_type == clingo.ast.ASTType.Literal and ast.head.atom == clingo.ast.BooleanConstant(0)
