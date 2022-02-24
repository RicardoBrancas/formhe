import clingo.ast

from asp.utils import is_fact, is_rule, is_integrity_constraint


def ast_callback(ast: clingo.ast.AST):
    print(ast)
    print('Is fact?', is_fact(ast))
    print('Is rule?', is_rule(ast))
    print('Is integrity constraint?', is_integrity_constraint(ast))
    print()


with open('examples/nqueens.lp') as f:
    content = f.read()

clingo.ast.parse_string(content, ast_callback)
