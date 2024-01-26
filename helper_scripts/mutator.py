import copy
import os.path
import random
from dataclasses import field
from pathlib import Path

import clingo.ast
from argparse_dataclass import dataclass
from clingo import SymbolType
from ordered_set import OrderedSet


@dataclass
class MutatorConfig:
    input_file: str = field(metadata={'args': ['input_file']})
    destination_path: str = field(metadata={'args': ['destination_path']})
    additional_definitions: list = field(metadata=dict(nargs='*', type=str), default_factory=list)
    additional_predicates: list = field(metadata=dict(nargs='*', type=str), default_factory=list)
    additional_numbers: list = field(metadata=dict(nargs='*', type=int), default_factory=lambda: [0, 1])
    iterations: int = 100
#     header: str = """%formhe-domain-predicates:assign/2
# %formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
# %formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
#
# """
#     header: str = """%formhe-domain-predicates:exec/4
# %formhe-instance-base64:dGltZXN0ZXAoMS4uOSkuCiNjb25zdCBsb3dlcmJvdW5kID0gOS4KI2NvbnN0IHVwcGVyYm91bmQgPSA5LgpkdXIoMSwxLDI7MiwxLDE7MSwyLDE7MiwyLDY7MSwzLDQ7MiwzLDE7KS4KbmV4dCgxLDEsMjsyLDEsMjsxLDIsMzsyLDIsMzspLg==
#
# """
#     header: str = """%formhe-domain-predicates:a/2 c/2 l/2 r/2 v/1
# %formhe-instance-base64:bm9kZSgxLi41KS4KZmVhdHVyZSgxLi41KS4KMyB7dihJKTogbm9kZShJKX0gMy4KMSB7ZDAoMSwgSSk7IGQwKDIsIEkpOyBkMCgzLCBJKTsgZDAoNCwgSSk7IGQwKDUsIEkpfSA1IDotIGMoSSwgMCksIHYoSSkuCjEge2QwKDEsIEkpOyBkMCgyLCBJKTsgZDEoMywgSSk7IGQxKDQsIEkpOyBkMCg1LCBJKX0gNSA6LSBjKEksIDApLCB2KEkpLgoxIHtkMCgxLCBJKTsgZDEoMiwgSSk7IGQwKDMsIEkpOyBkMCg0LCBJKTsgZDAoNSwgSSl9IDUgOi0gYyhJLCAxKSwgdihJKS4KMSB7ZDAoMSwgSSk7IGQxKDIsIEkpOyBkMSgzLCBJKTsgZDAoNCwgSSk7IGQwKDUsIEkpfSA1IDotIGMoSSwgMCksIHYoSSku
#
# """
#     header: str = """%formhe-domain-predicates:sel/1
# %formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K
#
# """
#     header: str = """%formhe-domain-predicates:sel/1
# %formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
#
# """
    header: str = """%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgyLCAzKS4KZSgzLCA0KS4KZSg0LCA1KS4KZSg1LCA2KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAxLgoKZSgxLCAzKS4KZSgyLCAzKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4=

"""
#     header: str = """%formhe-domain-predicates:set/2
# %formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K
#
# """

    min_changes: int = 1
    max_changes: int = 2
    p_replace_guards: float = 0.2
    p_replace_constants: float = 0.2
    p_replace_predicates: float = 0.2
    p_delete_constraint: float = 0.1


class ReplacementBag:

    def __init__(self):
        self.collector = ReplacementBagCollector()

    def getAggregateLowerBound(self, current):
        return self.getAggregateBound(current)

    def getAggregateUpperBound(self, current):
        return self.getAggregateBound(current)

    def getAggregateBound(self, current):
        return self.getNumericConstant(current)

    def getNumericConstant(self, current):
        options = OrderedSet(config.additional_numbers + config.additional_definitions)
        options = options.union(self.collector.numbers, self.collector.definitions)

        if current is not None:
            options.remove(current)

        choice = random.choice(options)

        if isinstance(choice, int):
            return clingo.Number(choice)
        else:
            return clingo.Function(choice)

    def getNamedConstant(self, current):
        options = OrderedSet(config.additional_definitions)
        options = options.union(self.collector.constants, self.collector.definitions)

        if current is not None:
            options.remove(current)

        choice = random.choice(options)

        return clingo.Function(choice)

    def getPredicateName(self, current):
        options = OrderedSet(config.additional_predicates)
        options = options.union(self.collector.functions)

        if current is not None:
            options.remove(current)

        choice = random.choice(options)

        return choice


class ReplacementBagCollector(clingo.ast.Transformer):

    def __init__(self):
        self.definitions = OrderedSet()
        self.functions = OrderedSet()
        self.variables = OrderedSet()
        self.numbers = OrderedSet()
        self.constants = OrderedSet()

    def visit_Definition(self, ast):
        self.definitions.add(ast.name)
        self.visit_children(ast)
        return ast

    def visit_Function(self, ast):
        self.functions.add(ast.name)
        self.visit_children(ast)
        return ast

    def visit_Variable(self, ast):
        self.variables.add(ast.name)
        self.visit_children(ast)
        return ast

    def visit_SymbolicTerm(self, ast):
        match ast.symbol.type:
            case clingo.symbol.SymbolType.Number:
                self.numbers.add(ast.symbol.number)
            case clingo.symbol.SymbolType.Function:
                self.constants.add(ast.symbol.name)
        self.visit_children(ast)
        return ast


class Transformer(clingo.ast.Transformer):

    def __init__(self, replacement_bag: ReplacementBag):
        self.bag = replacement_bag
        self.current_rule = -1
        self.deleted_lines = 0
        self.n_changes = 0
        self.changes_generate = []
        self.changes_test = []
        self.in_generate_rule = False

    def visit_Guard(self, ast):
        if random.random() < config.p_replace_guards and self.n_changes < config.max_changes:
            self.n_changes += 1
            if self.in_generate_rule:
                self.changes_generate.append(self.current_rule)
            else:
                self.changes_test.append(self.current_rule)
            if ast.term.ast_type == clingo.ast.ASTType.SymbolicTerm:
                return ast.update(term=ast.term.update(symbol=self.bag.getAggregateBound(ast.term.symbol.number if ast.term.symbol.type == SymbolType.Number else ast.term.symbol.name)))
            else:
                return ast.update(term=ast.term.update(symbol=self.bag.getAggregateBound(None)))
        return ast

    def visit_SymbolicTerm(self, ast):
        if ast.symbol.type == clingo.symbol.SymbolType.Function and random.random() < config.p_replace_constants and self.n_changes < config.max_changes:
            self.n_changes += 1
            if self.in_generate_rule:
                self.changes_generate.append(self.current_rule)
            else:
                self.changes_test.append(self.current_rule)
            return ast.update(symbol=self.bag.getNamedConstant(ast.term.symbol.number if ast.term.symbol.type == SymbolType.Number else ast.term.symbol.name))
        return ast

    def visit_Function(self, ast):
        if random.random() < config.p_replace_predicates and self.n_changes < config.max_changes:
            self.n_changes += 1
            if self.in_generate_rule:
                self.changes_generate.append(self.current_rule)
            else:
                self.changes_test.append(self.current_rule)
            return ast.update(name=self.bag.getPredicateName(ast.name))
        return ast

    def visit_Rule(self, ast):
        if ast.head and (ast.head.ast_type != clingo.ast.ASTType.Literal or ast.head.atom != clingo.ast.BooleanConstant(0)):
            self.in_generate_rule = True
        if random.random() < config.p_delete_constraint and self.n_changes < config.max_changes:
            self.deleted_lines += 1
            self.n_changes += 1
            self.changes_generate = [n for n in self.changes_generate if n != self.current_rule + 1]
            self.changes_generate = [n - 1 if n >= self.current_rule + 1 else n for n in self.changes_generate]
            self.changes_test = [n for n in self.changes_test if n != self.current_rule + 1]
            self.changes_test = [n - 1 if n >= self.current_rule + 1 else n for n in self.changes_test]
            self.in_generate_rule = False
            return None
        self.current_rule += 1
        updated_ast = ast.update(**self.visit_children(ast))
        self.in_generate_rule = False
        return updated_ast


def parse_string(*args, **kwargs):
    x = []
    clingo.ast.parse_string(*args, **kwargs, callback=x.append)
    return x


if __name__ == '__main__':
    config: MutatorConfig = MutatorConfig.parse_args()

    with open(config.input_file) as f:
        asts = parse_string(f.read())

    replacement_bag = ReplacementBag()

    for ast in asts:
        replacement_bag.collector.visit(ast)
        # print(ast)

    modified_asts = OrderedSet()

    i = 0
    while len(modified_asts) < config.iterations and i < 10 * config.iterations:
        i += 1
        transformer = Transformer(replacement_bag)

        m_asts = [copy.deepcopy(ast) for ast in asts]
        while transformer.n_changes == 0:
            transformer.current_rule = -1
            m_asts = [transformer.visit(ast) if ast is not None else ast for ast in m_asts]

        modified_asts.add((tuple(m_asts), transformer.deleted_lines, tuple(transformer.changes_generate), tuple(transformer.changes_test)))

    input_path = Path(config.input_file)
    destination_path = Path(config.destination_path)
    destination_path.mkdir(exist_ok=True)

    for i, (m_ast, n_deleted, n_generate, n_test) in enumerate(modified_asts):

        destination_file = destination_path / (str(i) + '.lp')
        relative_groundtruth = os.path.relpath(input_path, destination_path)

        with open(destination_file, 'w') as f:
            f.write('%formhe-groundtruth:' + relative_groundtruth + '\n')
            f.write('%formhe-selfeval-lines:' + ' '.join(map(str, OrderedSet(n_generate).union(OrderedSet(n_test)))) + '\n')
            f.write('%formhe-selfeval-deleted-lines:' + str(n_deleted) + '\n')
            f.write('%formhe-selfeval-changes-generate:' + ' '.join(map(str, n_generate)) + '\n')
            f.write('%formhe-selfeval-changes-generate-n:' + str(len(n_generate)) + '\n')
            f.write('%formhe-selfeval-changes-generate-n-unique:' + str(len(set(n_generate))) + '\n')
            f.write('%formhe-selfeval-changes-test:' + ' '.join(map(str, n_test)) + '\n')
            f.write('%formhe-selfeval-changes-test-n:' + str(len(n_test)) + '\n')
            f.write('%formhe-selfeval-changes-test-n-unique:' + str(len(set(n_test))) + '\n')
            f.write(config.header)

            for ast in m_ast:
                if ast is not None:
                    f.write(str(ast) + '\n')
                else:
                    f.write('%\n')

        print('Wrote', i)
