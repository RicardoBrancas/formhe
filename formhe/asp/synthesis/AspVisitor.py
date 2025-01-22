import logging
import re
from collections import Counter
from itertools import chain, pairwise
from typing import Union

import clingo.ast
from multiset import Multiset
from ordered_set import OrderedSet

from formhe.trinity.DSL import TyrellSpec, EnumType, Node, ApplyNode, AtomNode
from formhe.trinity.Visitor import Builder

logger = logging.getLogger('formhe.asp.visitor')


def list_nodes(node):
    if isinstance(node, list) or isinstance(node, tuple):
        return chain.from_iterable([list_nodes(n) for n in node])
    elif isinstance(node, ApplyNode):
        return chain([node.name], *[list_nodes(a) for a in node.children])
    elif isinstance(node, AtomNode):
        return [(node.type.name, node.data)]


def bag_nodes(node):
    counter = Counter()
    for elem in list_nodes(node):
        counter[elem] += 1
    result = Multiset()
    for elem, count in counter.items():
        if elem == 'empty':
            continue
        if not isinstance(elem, tuple):
            if match := re.fullmatch(r'p\d+/(\d+)', elem):
                result.add((f'p/{match.group(1)}', count))
            else:
                result.add(elem, count)
        else:
            if isinstance(elem[1], str) and re.fullmatch(r'X\d+', elem[1]):
                result.add(((elem[0], 'X'), count))
            else:
                result.add(elem, count)
    return result


class AspVisitor:

    def __init__(self, spec: TyrellSpec, free_vars: list = None, anonymize=False, anonymize_vars=False, anonymize_functions=False, domain_predicates=None):
        self.spec = spec
        self.builder = Builder(spec)
        if free_vars is not None:
            self.free_vars = OrderedSet(free_vars)
        else:
            self.free_vars = OrderedSet()
        if domain_predicates is not None:
            self.domain_predicates = OrderedSet(domain_predicates)
        else:
            self.domain_predicates = OrderedSet()
        if anonymize:
            anonymize_vars = True
            anonymize_functions = True
        self.anonymize_vars = anonymize_vars
        self.anonymize_functions = anonymize_functions
        self.anon_var_map = {}
        self.anon_var_count = 0
        self.anon_pred_map = {}
        self.anon_pred_count = 0

    def visit(self, ast: clingo.ast.AST, *args, **kwargs) -> Union[Node, tuple[Node, list[Node]]]:
        '''
        Dispatch to a visit method in a base class or visit and transform the
        children of the given AST if it is missing.
        '''
        attr = 'visit_' + str(ast.ast_type).replace('ASTType.', '')
        if hasattr(self, attr):
            return getattr(self, attr)(ast, *args, **kwargs)
        else:
            raise NotImplementedError(attr)

    def visit_BooleanConstant(self, boolean_constant):
        if boolean_constant.value == 0:
            return self.builder.make_enum('PBool', False)
        else:
            return self.builder.make_enum('PBool', True)

    def visit_SymbolicAtom(self, symbolic_atom):
        return self.visit(symbolic_atom.symbol)

    def visit_SymbolicTerm(self, symbolic_term):
        symbol = symbolic_term.symbol
        match symbol.type:
            case clingo.SymbolType.Number:
                Terminal: EnumType = self.spec.get_type('Terminal')
                if not symbol.number in Terminal.domain:
                    Terminal.domain.append(symbol.number)
                    self.spec._prod_spec.add_enum_production(Terminal, len(Terminal.domain) - 1, False)
                return self.builder.make_enum('Terminal', symbol.number)
            case clingo.SymbolType.Function:
                Terminal: EnumType = self.spec.get_type('Terminal')
                if not symbol.name in Terminal.domain:
                    Terminal.domain.append(symbol.name)
                    self.spec._prod_spec.add_enum_production(Terminal, len(Terminal.domain) - 1, False)
                return self.builder.make_enum('Terminal', symbol.name)
            case _:
                raise NotImplementedError()

    def visit_Literal(self, literal):
        if literal.sign == 0:
            return self.visit(literal.atom)
        else:
            return self.builder.make_apply('not', [self.visit(literal.atom)])

    def visit_Variable(self, variable):
        if not self.anonymize_vars:
            Terminal: EnumType = self.spec.get_type('Terminal')
            if variable.name not in Terminal.domain:
                Terminal.domain.append(variable.name)
                self.spec._prod_spec.add_enum_production(Terminal, len(Terminal.domain) - 1, False)
                self.free_vars.append(variable.name)
            return self.builder.make_enum('Terminal', variable.name)
        else:
            Terminal: EnumType = self.spec.get_type('Terminal')
            if variable.name not in self.anon_var_map:
                self.anon_var_map[variable.name] = f'X{self.anon_var_count}'
                self.anon_var_count += 1
            anon_var = self.anon_var_map[variable.name]
            if anon_var not in Terminal.domain:
                Terminal.domain.append(anon_var)
                self.spec._prod_spec.add_enum_production(Terminal, len(Terminal.domain) - 1, False)
                self.free_vars.append(anon_var)
            return self.builder.make_enum('Terminal', anon_var)

    def visit_Function(self, function):
        args = [self.visit(a) for a in function.arguments]
        if function.name:
            actual_fn_name = f'{function.name}/{len(args)}'
            if self.anonymize_functions and (function.name, len(args)) not in self.domain_predicates:
                if actual_fn_name not in self.anon_pred_map:
                    self.anon_pred_map[actual_fn_name] = f'p{self.anon_pred_count}/{len(args)}'
                    self.anon_pred_count += 1
                anon_pred = self.anon_pred_map[actual_fn_name]
                if not self.spec.get_function_production(anon_pred):
                    self.spec._prod_spec.add_func_production(anon_pred, self.spec.get_type('PBool'), [self.spec.get_type('Any')] * len(args))  # todo add predicates
                return self.builder.make_apply(anon_pred, args)
            else:
                if self.spec.get_function_production(actual_fn_name):
                    return self.builder.make_apply(actual_fn_name, args)
                else:
                    self.spec._prod_spec.add_func_production(actual_fn_name, self.spec.get_type('PBool'), [self.spec.get_type('Any')] * len(args))  # todo add predicates
                    return self.builder.make_apply(actual_fn_name, args)

        else:
            raise NotImplementedError()  # todo tuples

    def visit_Comparison(self, comparison):

        parts = []
        for a, guard in pairwise([comparison.term] + list(comparison.guards)):
            if a.ast_type == clingo.ast.ASTType.Guard:
                term = a.term
            else:
                term = a

            lhs = term
            comparison_op = guard.comparison
            rhs = guard.term
            match comparison_op:
                case clingo.ast.ComparisonOperator.Equal:
                    parts.append(self.builder.make_apply('eq', [self.visit(lhs), self.visit(rhs)]))
                    continue
                case clingo.ast.ComparisonOperator.LessThan:
                    parts.append(self.builder.make_apply('lt', [self.visit(lhs), self.visit(rhs)]))
                    continue
                case clingo.ast.ComparisonOperator.LessEqual:
                    parts.append(self.builder.make_apply('le', [self.visit(lhs), self.visit(rhs)]))
                    continue
                case clingo.ast.ComparisonOperator.GreaterThan:
                    parts.append(self.builder.make_apply('gt', [self.visit(lhs), self.visit(rhs)]))
                    continue
                case clingo.ast.ComparisonOperator.GreaterEqual:
                    parts.append(self.builder.make_apply('ge', [self.visit(lhs), self.visit(rhs)]))
                    continue
                case clingo.ast.ComparisonOperator.NotEqual:
                    if lhs.ast_type == clingo.ast.ASTType.Function and lhs.name == '' and \
                            rhs.ast_type == clingo.ast.ASTType.Function and rhs.name == '':  # tuple comparison
                        comparison_elements = []
                        for elem_a, elem_b in zip(lhs.arguments, rhs.arguments):
                            comparison_elements.append(self.builder.make_apply('neq', [self.visit(elem_a), self.visit(elem_b)]))
                        parts.append(self.builder.make_apply('and', comparison_elements))
                        continue
                    else:
                        parts.append(self.builder.make_apply('neq', [self.visit(lhs), self.visit(rhs)]))
                        continue
                case _:
                    raise NotImplementedError()

        if len(parts) == 1:
            return parts[0]
        else:
            res = self.builder.make_apply('and', parts)
            print(str(comparison))
            print(res)
            return res

    def visit_BinaryOperation(self, binary_operation):
        match binary_operation.operator_type:
            case clingo.ast.BinaryOperator.Plus:
                return self.builder.make_apply('add', [self.visit(binary_operation.left), self.visit(binary_operation.right)])
            case clingo.ast.BinaryOperator.Minus:
                return self.builder.make_apply('sub', [self.visit(binary_operation.left), self.visit(binary_operation.right)])
            case _:
                raise NotImplementedError()

    def visit_UnaryOperation(self, unary_operation):
        match unary_operation.operator_type:
            case clingo.ast.UnaryOperator.Negation:
                raise NotImplementedError()
            case clingo.ast.UnaryOperator.Minus:
                arg = self.visit(unary_operation.argument)
                if arg.type.name == 'Terminal':
                    return self.builder.make_apply('sub', [self.builder.make_enum('Terminal', 0), arg])
                else:
                    return self.builder.make_apply('classical_not', [arg])
            case clingo.ast.UnaryOperator.Absolute:
                return self.builder.make_apply('abs', [self.visit(unary_operation.argument)])
            case _:
                raise NotImplementedError()

    def visit_Disjunction(self, disjunction):
        elems = []
        for elem in disjunction.elements:
            if elem.ast_type != clingo.ast.ASTType.ConditionalLiteral:
                elems.append(self.visit(elem))
            elif elem.ast_type == clingo.ast.ASTType.ConditionalLiteral and not elem.condition:
                elems.append(self.visit(elem.literal))
            else:
                raise NotImplementedError()
        return self.builder.make_apply('or', elems)

    def visit_Guard(self, guard):
        match guard.comparison:
            case clingo.ast.ComparisonOperator.LessEqual:
                return self.visit(guard.term)
            case clingo.ast.ComparisonOperator.Equal:
                return self.visit(guard.term)
            case clingo.ast.ComparisonOperator.GreaterEqual:
                return self.visit(guard.term)
            case _:
                raise NotImplementedError()

    def visit_Aggregate(self, aggregate):
        left_guard, right_guard = self.extract_guards(aggregate)
        if len(aggregate.elements) == 1 and aggregate.elements[0].ast_type == clingo.ast.ASTType.ConditionalLiteral:
            elem = aggregate.elements[0]
            literal = self.visit(elem.literal)
            if len(elem.condition) == 1:
                condition = self.visit(elem.condition[0])
            elif len(elem.condition) == 0:
                condition = self.builder.make_enum('PBool', True)
            else:
                condition = self.builder.make_apply('and_', [self.visit(cond) for cond in elem.condition])

            if literal.name == "pool" and len(elem.condition) == 0:
                return self.builder.make_apply('aggregate_pool', [left_guard, literal, right_guard])
            else:
                return self.builder.make_apply('aggregate', [left_guard, literal, condition, right_guard])
        elif len(aggregate.elements) == 1 and aggregate.elements[0].ast_type == clingo.ast.ASTType.BodyAggregateElement and len(aggregate.elements[0].terms) == 1:
            elem = aggregate.elements[0]
            literal = self.visit(elem.terms[0])
            if len(elem.condition) == 1:
                condition = self.visit(elem.condition[0])
            elif len(elem.condition) == 0:
                condition = self.builder.make_enum('PBool', True)
            else:
                condition = self.builder.make_apply('and_', [self.visit(cond) for cond in elem.condition])

            return self.builder.make_apply('aggregate_term', [left_guard, literal, condition, right_guard])
        elif len(aggregate.elements) != 1:
            elems = []
            for elem in aggregate.elements:
                if elem.ast_type == clingo.ast.ASTType.ConditionalLiteral and not elem.condition:
                    elems.append(self.visit(elem.literal))
                else:
                    raise NotImplementedError()

            return self.builder.make_apply('aggregate_pool', [left_guard,
                                                              self.builder.make_apply('pool', elems),
                                                              right_guard])
        else:
            raise NotImplementedError()

    def extract_guards(self, aggregate):
        if aggregate.left_guard and aggregate.left_guard.comparison == clingo.ast.ComparisonOperator.LessEqual and \
                aggregate.right_guard and aggregate.right_guard.comparison == clingo.ast.ComparisonOperator.LessEqual:
            left_guard = self.visit(aggregate.left_guard)
            right_guard = self.visit(aggregate.right_guard)
        elif aggregate.left_guard and aggregate.left_guard.comparison == clingo.ast.ComparisonOperator.Equal and aggregate.right_guard is None:
            left_guard = self.visit(aggregate.left_guard)
            right_guard = left_guard
        elif aggregate.right_guard and aggregate.right_guard.comparison == clingo.ast.ComparisonOperator.Equal and aggregate.left_guard is None:
            right_guard = self.visit(aggregate.right_guard)
            left_guard = right_guard
        else:
            if aggregate.left_guard and aggregate.left_guard.comparison == clingo.ast.ComparisonOperator.LessEqual:
                left_guard = self.visit(aggregate.left_guard)
            elif aggregate.left_guard and aggregate.left_guard.comparison == clingo.ast.ComparisonOperator.GreaterEqual and not aggregate.right_guard:
                left_guard = self.builder.make_apply('empty', [])
                right_guard = self.visit(aggregate.left_guard)
                return left_guard, right_guard
            else:
                left_guard = self.builder.make_apply('empty', [])
            if aggregate.right_guard and aggregate.right_guard.comparison == clingo.ast.ComparisonOperator.LessEqual:
                right_guard = self.visit(aggregate.right_guard)
            elif aggregate.right_guard:
                raise NotImplementedError()
            else:
                right_guard = self.builder.make_apply('empty', [])
        return left_guard, right_guard

    def visit_BodyAggregate(self, body_aggregate):
        inner = clingo.ast.Aggregate(body_aggregate.location, body_aggregate.left_guard, body_aggregate.elements, body_aggregate.right_guard)
        return self.builder.make_apply('body_aggregate', [self.builder.make_enum('BodyAggregateFunc', clingo.ast.AggregateFunction(body_aggregate.function).name), self.visit(inner)])

    def visit_Pool(self, pool):
        return self.builder.make_apply('pool', [self.visit(a) for a in pool.arguments])

    def visit_Interval(self, interval):
        left = self.visit(interval.left)
        right = self.visit(interval.right)
        return self.builder.make_apply('interval', [left, right])

    def visit_Rule(self, rule):
        self.anon_var_map = {}
        self.anon_var_count = 0
        if rule.head.ast_type == clingo.ast.ASTType.Literal and rule.head.atom.ast_type == clingo.ast.ASTType.BooleanConstant and rule.head.atom.value == 0:
            head = self.builder.make_apply('empty', [])
        else:
            head = self.visit(rule.head)
        body = [self.visit(a) for a in rule.body]
        return self.builder.make_apply('stmt', [head, self.builder.make_apply('and', body)])

    def visit_Minimize(self, minimize):
        weight = self.visit(minimize.weight)
        priority = self.visit(minimize.priority)
        terms = self.builder.make_apply('and_', [self.visit(term) for term in minimize.terms])
        body = self.builder.make_apply('and_', [self.visit(l) for l in minimize.body])
        return self.builder.make_apply('minimize', [weight, priority, terms, body])

    def visit_Program(self, program):
        return self.builder.make_apply('empty', [])  # todo

    def visit_Definition(self, definition):
        return self.builder.make_apply('define', [
            self.builder.make_enum('Terminal', definition.name),
            self.visit(definition.value)
        ])  # todo

    def visit_ShowSignature(self, show_signature):
        return self.builder.make_apply('empty', [])  # todo
