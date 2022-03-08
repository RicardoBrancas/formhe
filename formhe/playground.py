import clingo.ast

from asp.instance import Instance
from asp.utils import is_fact, is_rule, is_integrity_constraint


def ast_callback(ast: clingo.ast.AST):
    print(ast)
    print('Is fact?', is_fact(ast))
    print('Is rule?', is_rule(ast))
    print('Is integrity constraint?', is_integrity_constraint(ast))
    print()


instance = Instance('buggy_instances/nqueens/0.lp')

instance.find_wrong_models(1000)

print("Cores")
instance.print_cores()
print("GT Cores")
instance.print_gt_cores()
# instance.print_answer_sets()

