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
        self.definitions = OrderedSet()
        self.predicates = {}
        self.predicates_typed = {}
        self.predicates_generated = OrderedSet()
        self.predicates_used = OrderedSet()
        self.rule_counter = 0
        self.skips = skips
        self.skipped = []
        self.not_skipped = []
        self.in_definition_block = False

    def visit_Definition(self, definition):
        self.rule_counter += 1
        if self.skips is not None and self.rule_counter - 1 in self.skips:
            self.skipped.append(definition)
            return None
        else:
            if not self.in_definition_block:
                self.not_skipped.append(definition)
            self.definitions.append(definition.name)
            return definition

    def visit_Function(self, function, in_skip=False, in_head=None, is_literal=None):
        arg_types = []
        add = True
        if function.name:
            self.predicates[f'{function.name}/{len(function.arguments)}'] = len(function.arguments)

        # print(function, in_skip, in_head, is_literal)

        if not in_skip and in_head and (is_literal or is_literal is None):
            self.predicates_generated.append(f'{function.name}/{len(function.arguments)}')
        elif not in_skip and (not in_head or (in_head and not is_literal)):
            self.predicates_used.append(f'{function.name}/{len(function.arguments)}')

        for arg in function.arguments:
            if arg.ast_type == clingo.ast.ASTType.SymbolicTerm:
                arg_types.append(arg.symbol.type)
            else:
                add = False

            if arg.ast_type == clingo.ast.ASTType.SymbolicTerm and arg.symbol.type == clingo.SymbolType.Function and len(
                    arg.symbol.arguments) == 0:
                self.constants.append(arg.symbol.name)
        if add:
            self.predicates_typed[f'{function.name}/{len(function.arguments)}'] = arg_types
        return function

    def visit_ConditionalLiteral(self, conditional_literal, **kwargs):
        conditional_literal.update(literal=self._dispatch(conditional_literal.literal, is_literal=True, **kwargs),
                                   condition=self._dispatch(conditional_literal.condition, is_literal=False, **kwargs))
        return conditional_literal

    def visit_Rule(self, rule):
        self.rule_counter += 1
        try:
            if rule.head.atom.symbol.name == "formhe_definition_begin":
                self.in_definition_block = True
                return None
            if rule.head.atom.symbol.name == "formhe_definition_end":
                self.in_definition_block = False
                return None
        except:
            pass

        if self.skips is not None and self.rule_counter - 1 in self.skips:
            self.skipped.append(rule)
            rule.update(head=self._dispatch(rule.head, in_skip=True, in_head=True), body=self._dispatch(rule.body, in_skip=True, in_head=False))
            return None
        else:
            if not self.in_definition_block:
                self.not_skipped.append(rule)
            # rule.update(**self.visit_children(rule))
            rule.update(head=self._dispatch(rule.head, in_head=True), body=self._dispatch(rule.body, in_head=False))
            return rule


class Instrumenter(clingo.ast.Transformer):

    def __init__(self):
        self.counter = 0
        self.rule_counter = 0
        self.rules = []
        self.original_rules = []
        self.relaxation_functions = []
        self.relaxations_function_map = {}
        self.disabled = False

    def visit_Definition(self, definition):
        self.rule_counter += 1
        if not self.disabled:
            relaxation_function = clingo.Function(f"_instrumenter", [clingo.Number(self.counter)])
            self.relaxation_functions.append(relaxation_function)
            self.relaxations_function_map[relaxation_function] = self.rule_counter - 1
            self.original_rules.append(str(definition))
            self.counter += 1
            self.rules.append(definition)
            return definition
        else:
            return definition

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
            self.original_rules.append(str(rule))
            rule.body.append(clingo.ast.Literal(rule.location,
                                                clingo.ast.Sign.NoSign,
                                                clingo.ast.SymbolicAtom(
                                                    clingo.ast.Function(rule.location,
                                                                        f"_instrumenter",
                                                                        [clingo.ast.SymbolicTerm(rule.location, clingo.Number(self.counter))],
                                                                        0))))
            self.counter += 1
            self.rules.append(rule)
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
