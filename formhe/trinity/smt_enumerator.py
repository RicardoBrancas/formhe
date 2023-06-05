import itertools
import logging
from abc import ABC
from collections import namedtuple
from functools import cached_property
from typing import Callable, TypeVar, Union, NamedTuple, Any

# import cvc5.pythonic
import z3
from ordered_set import OrderedSet

import formhe.trinity.DSL
import runhelper
from formhe.trinity.DSL import TyrellSpec, ApplyNode, AtomNode
from formhe.trinity.DSL.type import Type
from formhe.utils import config

logger = logging.getLogger('formhe.asp.enumerator')

ExprType = TypeVar('ExprType')

Solver = Union[z3.Solver]
ExprRef = Union[z3.ExprRef]


class AST:
    def __init__(self):
        self.head = None


class Node:
    def __init__(self, enumerator: 'SmtEnumerator', id: int, depth: int, is_leaf: bool, n_child_indicator: int = None):
        self.id = id
        self.depth = depth
        self.children: list['Node'] = []
        self.is_leaf = is_leaf
        self.var = self._create_var(enumerator, n_child_indicator)
        self.semantic_vars = self._create_semantic_vars(enumerator)
        self.bound = False

    def _create_var(self, enumerator: 'SmtEnumerator', n_child_indicator: int = None) -> ExprRef:
        var = enumerator.create_variable(f'node_{self.id}', enumerator.smt_namespace.Int)
        enumerator.create_assertion(var >= 0)
        enumerator.create_assertion(var < enumerator.spec.num_productions())
        ctr = []
        for function in enumerator.spec.get_function_productions():
            if self.is_leaf and function.name != 'empty':
                ctr.append(var != function.id)
            if n_child_indicator is not None and n_child_indicator < len(function.rhs):
                ctr.append(var != function.id)
        enumerator.create_assertion(enumerator.smt_namespace.And(*ctr))

        return var

    def _create_semantic_vars(self, enumerator: 'SmtEnumerator') -> dict[Type, ExprRef]:
        sem_vars = {
            enumerator.spec.get_type('Bool'): enumerator.create_variable(f'node_{self.id}_Bool', enumerator.smt_namespace.Bool),
            enumerator.spec.get_type('Terminal'): enumerator.create_variable(f'node_{self.id}_Int', enumerator.smt_namespace.Int),
        }

        return sem_vars

    def collect_vars(self) -> list[ExprRef]:
        vars = [self.var]
        for child in self.children:
            vars += child.collect_vars()
        return vars

    def collect_nodes(self) -> list['Node']:
        nodes = [self]
        for child in self.children:
            nodes += child.collect_nodes()
        return nodes

    def bind(self, production_id: int, enumerator: 'SmtEnumerator'):
        self.bound = True
        enumerator.create_assertion(self.var == production_id, f'node_{self.id}_binding', True)

    def __repr__(self) -> str:
        return f'Node({self.id}, vars={repr(self.collect_vars())}, semantic_variables=[{", ".join(map(lambda t: t.name, self.semantic_vars.keys()))}], bound={self.bound})'

    @staticmethod
    def collect_vars_pairwise(node_a, node_b):
        vars = [(node_a.var, node_b.var)]
        for child_a, child_b in zip(node_a.children, node_b.children):
            vars += Node.collect_vars_pairwise(child_a, child_b)
        return vars


class PresetStatement(NamedTuple):
    has_head: bool
    head: Any
    body: list[Any]

    def __repr__(self) -> str:
        r = ''
        if self.has_head:
            if self.head is not None:
                r += str(self.head)
            else:
                r += '?'
        if len(self.body) > 0:
            r += ' :- '
            r += ', '.join(map(lambda x: '?' if x is None else str(x), self.body))
        r += '.'
        return r


class Statement:

    def __init__(self, enumerator: 'SmtEnumerator', tree_depth: int, preset_statement: PresetStatement):
        self.id = enumerator.statement_counter
        enumerator.statement_counter += 1
        self.nodes = []
        self.head_nodes = []
        self.body_nodes = []

        if preset_statement.has_head:
            if preset_statement.head is None:
                self.head = self.create_tree(enumerator, tree_depth, enumerator.max_children_with_aggregate, is_head=True)
            else:
                self.head = self.create_bound_tree(enumerator, preset_statement.head, is_head=True)
        else:
            self.head = self.create_bound_tree(enumerator, None, is_head=True)

        self.body = []
        for preset_atom in preset_statement.body:
            if preset_atom is not None:
                self.body.append(self.create_bound_tree(enumerator, preset_atom, is_body=True))
            else:
                self.body.append(self.create_tree(enumerator, tree_depth, enumerator.max_children_except_aggregate, is_body=True))

        self.create_head_output_constraints(enumerator, head_nullable=not preset_statement.has_head)
        self.create_body_output_constraints(enumerator)
        for root in self.body:
            self.create_aggregate_restrictions_constraints(enumerator, root)
        for child in self.head.children:
            self.create_aggregate_restrictions_constraints(enumerator, child)
        self.create_children_constraints(enumerator)
        if config.get().block_constant_expressions:
            self.create_no_constant_sub_expressions_constraints(enumerator)
        self.create_head_empty_or_non_constant_constraints(enumerator)
        self.create_no_dont_care_in_head_constraints(enumerator)
        self.create_no_constant_aggregate_constraints(enumerator)

    def finish_init(self, enumerator: 'SmtEnumerator'):
        self.create_lexicographic_order_constraints(enumerator)

    def create_tree(self, enumerator: 'SmtEnumerator', depth: int, max_children: int, current_depth: int = 1, is_body: bool = False, is_head: bool = False) -> Node:
        """ Builds a K-tree that will contain the program
            The recursive call changes the max_children parameter to force exclude aggregate,
            since aggregate can only occur as the top element in the tree
        """
        node = Node(enumerator, enumerator.node_counter, current_depth, current_depth == depth, max_children)
        self.nodes.append(node)
        if is_body:
            self.body_nodes.append(node)
        if is_head:
            self.head_nodes.append(node)
        enumerator.node_counter += 1
        if current_depth < depth:
            for i in range(max_children):
                node.children.append(self.create_tree(enumerator, depth, enumerator.max_children_except_aggregate, current_depth + 1, is_body, is_head))
        return node

    def create_bound_tree(self, enumerator: 'SmtEnumerator', ast_node: formhe.trinity.DSL.Node, is_body: bool = False, is_head: bool = False) -> Node:
        node = Node(enumerator, enumerator.node_counter, 0, False, enumerator.max_children_except_aggregate)
        self.nodes.append(node)
        if is_body:
            self.body_nodes.append(node)
        if is_head:
            self.head_nodes.append(node)
        enumerator.node_counter += 1
        if ast_node is None:
            node.bind(0, enumerator)
        elif isinstance(ast_node, ApplyNode):
            node.bind(ast_node.production.id, enumerator)
            for child in ast_node.children:
                node.children.append(self.create_bound_tree(enumerator, child, is_body, is_head))
        elif isinstance(ast_node, AtomNode):
            node.bind(ast_node.production.id, enumerator)

        return node

    def create_head_output_constraints(self, enumerator: 'SmtEnumerator', head_nullable: bool):
        '''The output production matches the output type'''
        output_tys = OrderedSet()
        if isinstance(enumerator.spec.output, tuple):
            for ty in enumerator.spec.output:
                output_tys.append(ty)
        else:
            output_tys.append(self.spec.output)
        output_tys.append(enumerator.spec.get_type('Aggregate'))
        if head_nullable:
            output_tys.append(enumerator.spec.get_type('Empty'))
        ctr = []
        for ty in output_tys:
            for p in enumerator.spec.get_productions_with_lhs(ty):
                ctr.append(self.head.var == p.id)
        enumerator.create_assertion(enumerator.smt_namespace.Or(*ctr))

    def create_body_output_constraints(self, enumerator: 'SmtEnumerator'):
        '''The output production matches the output type'''
        output_tys = OrderedSet()
        if isinstance(enumerator.spec.output, tuple):
            for ty in enumerator.spec.output:
                output_tys.append(ty)
        else:
            output_tys.append(enumerator.spec.output)
        for root in self.body:
            ctr = []
            for ty in output_tys:
                for p in enumerator.spec.get_productions_with_lhs(ty):
                    ctr.append(root.var == p.id)
            enumerator.create_assertion(enumerator.smt_namespace.Or(*ctr))

    def create_aggregate_restrictions_constraints(self, enumerator: 'SmtEnumerator', root: Node):
        aggregate = enumerator.spec.get_function_production_or_raise('aggregate')
        constraint = []
        for var in root.collect_vars():
            constraint.append(var != aggregate.id)
        enumerator.create_assertion(enumerator.smt_namespace.And(*constraint))

    def create_children_constraints(self, enumerator: 'SmtEnumerator'):
        for node in self.nodes:
            if node.children:
                for p in enumerator.spec.productions():
                    assert len(node.children) > 0
                    for child_i, child in enumerate(node.children):
                        ctr = []
                        child_types = ('Empty',)
                        if p.is_function() and child_i < len(p.rhs):
                            child_types = p.rhs[child_i]
                            if not isinstance(child_types, tuple):
                                child_types = (child_types,)
                        for child_type in child_types:
                            child_type = str(child_type)
                            if child_type != 'Any':
                                for t in enumerator.spec.get_productions_with_lhs(child_type):
                                    ctr.append(child.var == t.id)
                            else:
                                for t in enumerator.spec.productions():
                                    if t.lhs.name != 'Empty':
                                        ctr.append(child.var == t.id)
                        enumerator.create_assertion(enumerator.smt_namespace.Implies(node.var == p.id, enumerator.smt_namespace.Or(*ctr)))

    def create_head_empty_or_non_constant_constraints(self, enumerator: 'SmtEnumerator'):
        lhs_ctr = []
        for var in self.head.collect_vars():
            lhs_ctr.append(var == 0)
        lhs_ctr = enumerator.smt_namespace.And(*lhs_ctr)
        rhs_ctr = []
        for production in enumerator.spec.productions():
            if (not production.is_constant and production.is_enum()) or (production.is_function() and production.name == 'interval'):
                # print(production)
                for var in self.head.collect_vars():
                    rhs_ctr.append(var == production.id)
        rhs_ctr = enumerator.smt_namespace.Or(*rhs_ctr)
        enumerator.create_assertion(enumerator.smt_namespace.Or(lhs_ctr, rhs_ctr), f'{self.id}_head_empty_or_non_const')

    def create_no_dont_care_in_head_constraints(self, enumerator: 'SmtEnumerator'):
        dont_care_prod = enumerator.spec.get_enum_production(enumerator.spec.get_type('Terminal'), '_')
        ctr = []
        if len(self.head.children) >= 4:
            for var in self.head.children[0].collect_vars() + self.head.children[1].collect_vars() + self.head.children[3].collect_vars():
                ctr.append(var != dont_care_prod.id)
            for var in self.head.children[2].collect_vars():
                ctr.append(enumerator.smt_namespace.Or(self.head.var == enumerator.spec.get_function_production('aggregate').id, var != dont_care_prod.id))
        else:
            for var in self.head.collect_vars():
                ctr.append(var != dont_care_prod.id)
        enumerator.create_assertion(enumerator.smt_namespace.And(*ctr), f'no_dont_care_in_head_{self.id}')

    def create_no_constant_sub_expressions_constraints(self, enumerator: 'SmtEnumerator'):
        """For each subtree, either the tree is empty, or at least one non-const production is used"""
        subtrees = []
        for node in self.nodes:
            if node.children:
                subtrees.append(node.collect_vars())

        for subtree in subtrees:
            ctr = []
            ctr_2 = []
            for node in subtree:
                ctr_2.append(node == 0)
                for prod in enumerator.spec.productions():
                    if not prod.is_constant:
                        ctr.append(node == prod.id)
            enumerator.create_assertion(enumerator.smt_namespace.Or(*ctr, enumerator.smt_namespace.And(*ctr_2)))

    def create_no_constant_aggregate_constraints(self, enumerator: 'SmtEnumerator'):
        agg_production = enumerator.spec.get_function_production('aggregate')

        ctr = []
        if len(self.head.children) >= 4:
            for var in self.head.children[1].collect_vars():
                for prod in enumerator.spec.productions():
                    if not prod.is_constant and prod.is_enum():
                        ctr.append(var == prod.id)
            enumerator.create_assertion(enumerator.smt_namespace.Or(self.head.var != agg_production.id,
                                                                    enumerator.smt_namespace.Or(*ctr)))

    def create_no_unsafe_vars_constraints(self, enumerator: 'SmtEnumerator', free_vars: list):
        self.create_no_unsafe_vars_body_constraints(enumerator, free_vars)
        self.create_no_unsafe_vars_head_constraints(enumerator, free_vars)

    def create_no_unsafe_vars_body_constraints(self, enumerator: 'SmtEnumerator', free_vars: list):
        if config.get().allow_unsafe_vars:
            return

        aux_vars = {v: enumerator.create_variable(f'var_used_body_{v}_stmt_{self.id}', enumerator.smt_namespace.Bool) for v in free_vars}

        for var in free_vars:
            var_production = enumerator.spec.get_enum_production(enumerator.spec.get_type('Terminal'), var)
            ctr = []
            for root in self.body_nodes:
                if root.children:
                    sub_ctr = []
                    for predicate_name in enumerator.predicate_names:
                        production = enumerator.spec.get_function_production_or_raise(predicate_name)
                        sub_ctr.append(root.var == production.id)
                    sbt_ctr = enumerator.smt_namespace.And(enumerator.smt_namespace.Or(*sub_ctr),
                                                           enumerator.smt_namespace.Or(*[child.var == var_production.id for child in root.children]))
                    ctr.append(sbt_ctr)

            enumerator.create_assertion(aux_vars[var] == enumerator.smt_namespace.Or(ctr))

            for node in self.body_nodes:
                enumerator.create_assertion(enumerator.smt_namespace.Implies(enumerator.smt_namespace.Not(aux_vars[var]), node.var != var_production.id))

            for node in self.head_nodes:
                enumerator.create_assertion(enumerator.smt_namespace.Or(self.head.var == enumerator.spec.get_function_production('aggregate').id,
                                                                        enumerator.smt_namespace.Implies(enumerator.smt_namespace.Not(aux_vars[var]), node.var != var_production.id)))

            if len(self.head.children) >= 4:
                for node in self.head.children[0].collect_nodes() + self.head.children[3].collect_nodes():
                    enumerator.create_assertion(enumerator.smt_namespace.Or(self.head.var != enumerator.spec.get_function_production('aggregate').id,
                                                                            enumerator.smt_namespace.Implies(enumerator.smt_namespace.Not(aux_vars[var]), node.var != var_production.id)))

    def create_no_unsafe_vars_head_constraints(self, enumerator: 'SmtEnumerator', free_vars: list):
        if config.get().allow_unsafe_vars or len(self.head.children) < 4:
            return

        aux_vars = {v: enumerator.create_variable(f'var_used_head_{v}_stmt_{self.id}', enumerator.smt_namespace.Bool) for v in free_vars}
        agg_production = enumerator.spec.get_function_production('aggregate')

        for var in free_vars:
            var_production = enumerator.spec.get_enum_production(enumerator.spec.get_type('Terminal'), var)
            ctr = []
            for root in self.body_nodes:
                if root.children:
                    sub_ctr = []
                    for predicate_name in enumerator.predicate_names:
                        production = enumerator.spec.get_function_production_or_raise(predicate_name)
                        sub_ctr.append(root.var == production.id)
                    sbt_ctr = enumerator.smt_namespace.And(enumerator.smt_namespace.Or(*sub_ctr),
                                                           enumerator.smt_namespace.Or(*[child.var == var_production.id for child in root.children]))
                    ctr.append(sbt_ctr)

            for root in self.head.children[2].collect_nodes():
                if root.children:
                    sub_ctr = []
                    for predicate_name in enumerator.predicate_names:
                        production = enumerator.spec.get_function_production_or_raise(predicate_name)
                        sub_ctr.append(root.var == production.id)
                    sbt_ctr = enumerator.smt_namespace.And(self.head.var == agg_production.id,
                                                           enumerator.smt_namespace.Or(*sub_ctr),
                                                           enumerator.smt_namespace.Or(*[child.var == var_production.id for child in root.children]))
                    ctr.append(sbt_ctr)

            enumerator.create_assertion(aux_vars[var] == enumerator.smt_namespace.Or(ctr))

            for node in self.head.children[1].collect_nodes():
                enumerator.create_assertion(enumerator.smt_namespace.Or(self.head.var != agg_production.id,
                                                                        enumerator.smt_namespace.Implies(enumerator.smt_namespace.Not(aux_vars[var]), node.var != var_production.id)))

    def create_lexicographic_order_constraints(self, enum: 'SmtEnumerator', root: Node = None):
        if root is None:
            if len(self.body) > 1:
                fake_node = FakeNode(enum.commutative_prod_ids[0], [r for r in self.body if not r.bound], False)
                enum.create_assertion(self.create_lexicographic_order_constraints(enum, fake_node), f'lexicographic_body')
            elif len(self.body) > 0 and not self.body[0].bound:
                enum.create_assertion(self.create_lexicographic_order_constraints(enum, self.body[0]), 'lexicographic_body')

            enum.create_assertion(self.create_lexicographic_order_constraints(enum, self.head), f'lexicographic_head')

        else:
            if root.children:
                root_is_commutative = []
                for p in enum.commutative_prod_ids:
                    root_is_commutative.append(root.var == p)
                root_is_commutative = enum.smt_namespace.Or(*root_is_commutative)

                lex_constraints = []
                for child_a, child_b in itertools.pairwise(root.children):
                    var_pairs = Node.collect_vars_pairwise(child_a, child_b)

                    lex_constraints.append(var_pairs[0][0] <= var_pairs[0][1])

                    for i, (var_a, var_b) in enumerate(var_pairs[1:]):
                        lex_constraints_tmp = []
                        for var_1, var_2 in var_pairs[:i]:
                            lex_constraints_tmp.append(var_1 == var_2)
                        lex_constraints.append(enum.smt_namespace.Implies(enum.smt_namespace.And(*lex_constraints_tmp), var_a <= var_b))

                root_ctr = enum.smt_namespace.Implies(root_is_commutative, enum.smt_namespace.And(*lex_constraints))
                return enum.smt_namespace.And(root_ctr, *[self.create_lexicographic_order_constraints(enum, child) for child in root.children])
            else:
                return True

    def collect_vars(self):
        return itertools.chain(self.head.collect_vars(), *[root.collect_vars() for root in self.body])

    def __repr__(self):
        return f'Statement(head={self.head}, body={repr(self.body)})'


FakeNode = namedtuple('FakeNode', ['var', 'children', 'bound'])


# FIXME: Currently this enumerator requires an "Empty" production to function properly
class SmtEnumerator(ABC):
    # productions that are leaf
    leaf_productions = []

    # map from internal k-tree to nodes of program
    program2tree = {}

    def __init__(self, spec: TyrellSpec, depth=None, predicates_names=None, cores=None, free_vars=None, preset_statements=None, strict_minimum_depth=True, free_predicates=None,
                 force_generate_predicates=None):
        if predicates_names is None:
            predicates_names = []
        if preset_statements is None:
            preset_statements = []

        self.solver: Solver = self.init_solver()

        self.predicate_names = predicates_names
        self.cores = cores
        self.free_vars = free_vars

        self.var_counter = 0
        self.assertion_counter = 0
        self.template_var_counter = 0
        self.node_counter = 0
        self.statement_counter = 0

        self.leaf_productions = []
        self.program2tree = {}
        self.spec = spec
        if depth <= 0:
            raise ValueError('Depth cannot be non-positive: {}'.format(depth))
        self.depth = depth
        self.var_counter = 0

        self.statements = [Statement(self, depth, preset_statement) for preset_statement in preset_statements]

        # print(self.statements)

        # for node in self.nodes:
        #     print('   ' * (node.depth - 1) + str(node))
        self.model = None
        # self.create_input_constraints()

        self.commutative_prod_ids = []

        self.resolve_predicates()

        for statement in self.statements:
            statement.finish_init(self)

        if strict_minimum_depth:
            self.create_max_depth_used_constraints()
        # self.create_semantic_constraints()

        if not config.get().allow_not_generated_predicates:
            self.create_predicate_usage_constraints(free_predicates)
            self.create_predicate_force_generate_constraints(force_generate_predicates)

        self.blocking_template = self.blocking_template_compute()

        # print(self.spec._prod_spec)

        self.and_production = self.spec.get_function_production('and')
        self.stmt_and_production = self.spec.get_function_production('stmt_and')
        self.stmt_production = self.spec.get_function_production('stmt')
        self.has_enumerated = False

    def init_solver(self):
        raise NotImplementedError()

    @cached_property
    def smt_namespace(self) -> z3:
        raise NotImplementedError()

    def __del__(self):
        self.log()

    def log(self):
        logger.info('enum.smt.vars=%d', self.var_counter)
        logger.info('enum.smt.constraints=%d', self.assertion_counter)

    def create_variable(self, name: str, type: 'Callable[[str, ...], ExprType]', *args) -> ExprType:
        self.var_counter += 1
        return type(name, *args)

    def create_assertion(self, expr: z3.ExprRef, name: str = None, track: bool = False, debug_print: bool = False) -> None:
        # print(self.smt_namespace.simplify(expr))
        if debug_print:
            logger.info((name if name else '') + '\n' + str(expr))
        self.assertion_counter += 1
        if not config.get().no_enumerator_debug and track and name is not None:
            self.solver.assert_and_track(expr, name)
        else:
            self.solver.add(expr)

    # def create_input_constraints(self):
    #     '''Each input will appear at least once in the program'''
    #     input_productions = self.spec.get_param_productions()
    #     for x in range(0, len(input_productions)):
    #         ctr = []
    #         for node in self.nodes:
    #             ctr.append(node.var == input_productions[x].id)
    #         self.solver.add(self.smt_namespace.Or(*ctr))

    def create_no_unsafe_vars_constraints(self, free_vars: list):
        # print('No unsafe vars', free_vars)
        for stmt, stmt_free_vars in zip(self.statements, free_vars):
            stmt.create_no_unsafe_vars_constraints(self, stmt_free_vars)

    def create_force_var_usage_constraints(self, unsafe_vars: list):
        # print('Force var usage', unsafe_vars)
        if config.get().allow_unsafe_vars:
            return

        for stmt, stmt_unsafe_vars in zip(self.statements, unsafe_vars):
            for var in stmt_unsafe_vars:
                var_production = self.spec.get_enum_production(self.spec.get_type('Terminal'), var)
                ctr = []
                for node in stmt.body_nodes:
                    ctr.append(node.var == var_production.id)
                self.create_assertion(self.smt_namespace.Or(*ctr))

    def create_max_depth_used_constraints(self):
        ctr = []
        for smt in self.statements:
            for node in smt.nodes:
                if node.is_leaf and not node.bound and node.depth == self.depth:
                    ctr.append(node.var != 0)
        self.create_assertion(self.smt_namespace.Or(*ctr), 'max_depth_used', False, False)

    def create_predicate_usage_constraints(self, predicates: list[str]):
        aggregate_prod = self.spec.get_function_production('aggregate')
        for predicate in predicates:
            predicate_prod = self.spec.get_function_production(predicate)

            lhs = []
            for stmt in self.statements:
                for node in stmt.body_nodes:
                    lhs.append(node.var == predicate_prod.id)
            lhs = self.smt_namespace.Or(*lhs)

            rhs = []
            for stmt in self.statements:
                if len(stmt.head.children) >= 4:
                    rhs.append(self.smt_namespace.Or(stmt.head.var == predicate_prod.id, self.smt_namespace.And(stmt.head.var == aggregate_prod.id, stmt.head.children[1].var == predicate_prod.id)))
                else:
                    rhs.append(stmt.head.var == predicate_prod.id)
            rhs = self.smt_namespace.Or(*rhs)

            self.create_assertion(self.smt_namespace.Implies(lhs, rhs), f'predicate_usage_{predicate}', False, False)

    def create_predicate_force_generate_constraints(self, predicates: list[str]):
        aggregate_prod = self.spec.get_function_production('aggregate')
        for predicate in predicates:
            predicate_prod = self.spec.get_function_production(predicate)

            rhs = []
            for stmt in self.statements:
                if len(stmt.head.children) >= 4:
                    rhs.append(self.smt_namespace.Or(stmt.head.var == predicate_prod.id, self.smt_namespace.And(stmt.head.var == aggregate_prod.id, stmt.head.children[1].var == predicate_prod.id)))
                else:
                    rhs.append(stmt.head.var == predicate_prod.id)
            rhs = self.smt_namespace.Or(*rhs)

            self.create_assertion(rhs, f'predicate_force_{predicate}', False, False)

    # def create_semantic_constraints(self):
    #     if self.cores is None or self.cores == [] or config.get().no_semantic_constraints:
    #         return
    #
    #     Bool: EnumType = self.spec.get_type('Bool')
    #     Terminal: EnumType = self.spec.get_type('Terminal')
    #
    #     free_vars_semantic_vars = defaultdict(dict)
    #
    #     for var in self.free_vars:
    #         free_vars_semantic_vars[var][Bool] = self.create_variable(f'var_{var}_Bool', self.smt_namespace.Bool)
    #         free_vars_semantic_vars[var][Terminal] = self.create_variable(f'var_{var}_Int', self.smt_namespace.Int)
    #
    #     for i_node, node in enumerate(self.nodes):
    #         self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('empty').id,
    #                                                          self.smt_namespace.And(
    #                                                              node.semantic_vars[Int] == 0,
    #                                                              self.smt_namespace.Not(node.semantic_vars[Bool])
    #                                                          )), f'semantic_constraint_empty_{i_node}', True)
    #
    #         if node.children:
    #             self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('and').id,
    #                                                              self.smt_namespace.And(
    #                                                                  node.semantic_vars[Int] == 0,
    #                                                                  node.semantic_vars[Bool] == self.smt_namespace.And([c.semantic_vars[Bool] for c in node.children])
    #                                                              )), f'semantic_constraint_and_{i_node}', True)
    #
    #             self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('add').id,
    #                                                              self.smt_namespace.And(
    #                                                                  node.semantic_vars[Int] == node.children[0].semantic_vars[Int] + node.children[1].semantic_vars[Int],
    #                                                                  self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                              )), f'semantic_constraint_add_{i_node}', True)
    #
    #             self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('sub').id,
    #                                                              self.smt_namespace.And(
    #                                                                  node.semantic_vars[Int] == (node.children[0].semantic_vars[Int] - node.children[1].semantic_vars[Int]),
    #                                                                  self.smt_namespace.Not(node.semantic_vars[Bool])
    #                                                              )), f'semantic_constraint_sub_{i_node}', True)
    #
    #             self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('mul').id,
    #                                                              self.smt_namespace.And(
    #                                                                  node.semantic_vars[Int] == node.children[0].semantic_vars[Int] * node.children[1].semantic_vars[Int],
    #                                                                  self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                              )), f'semantic_constraint_mul_{i_node}', True)
    #
    #             self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('div').id,
    #                                                              self.smt_namespace.And(
    #                                                                  node.semantic_vars[Int] == node.children[0].semantic_vars[Int] / node.children[1].semantic_vars[Int],
    #                                                                  self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                              )), f'semantic_constraint_div_{i_node}', True)
    #
    #             self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('abs').id,
    #                                                              self.smt_namespace.And(
    #                                                                  node.semantic_vars[Int] == self.smt_namespace.Abs(node.children[0].semantic_vars[Int]),
    #                                                                  self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                              )), f'semantic_constraint_abs_{i_node}', True)
    #
    #             self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('eq').id,
    #                                                              self.smt_namespace.And(
    #                                                                  node.semantic_vars[Bool] == (node.children[0].semantic_vars[Int] == node.children[1].semantic_vars[Int]),
    #                                                                  node.semantic_vars[Bool] == (node.children[0].semantic_vars[Bool] == node.children[1].semantic_vars[Bool])
    #                                                              )), f'semantic_constraint_eq_{i_node}', True)
    #
    #             self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('neq').id,
    #                                                              self.smt_namespace.Or(
    #                                                                  node.semantic_vars[Bool] == (node.children[0].semantic_vars[Int] != node.children[1].semantic_vars[Int]),
    #                                                                  node.semantic_vars[Bool] == (node.children[0].semantic_vars[Bool] != node.children[1].semantic_vars[Bool])
    #                                                              )), f'semantic_constraint_neq_{i_node}', True)
    #
    #         for prod in self.spec.get_productions_with_lhs(Int):
    #             if prod.is_enum() and prod.is_constant:
    #                 self.create_assertion(self.smt_namespace.Implies(node.var == prod.id,
    #                                                                  self.smt_namespace.And(
    #                                                                      node.semantic_vars[Int] == prod._get_rhs(),
    #                                                                      self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                                  )), f'semantic_constraint_int_const_{prod.id}_{i_node}', True)
    #         for var in self.free_vars:
    #             prod = self.spec.get_enum_production(Int, var)
    #             self.create_assertion(self.smt_namespace.Implies(node.var == prod.id,
    #                                                              self.smt_namespace.And(
    #                                                                  node.semantic_vars[Int] == free_vars_semantic_vars[var][Int],
    #                                                                  node.semantic_vars[Bool] == free_vars_semantic_vars[var][Bool]
    #                                                              )), f'semantic_constraint_var_{var}_{i_node}', True)
    #
    #         for prod in self.spec.get_productions_with_lhs(Bool):
    #             if prod.is_enum() and prod.is_constant:
    #                 self.create_assertion(self.smt_namespace.Implies(node.var == prod.id,
    #                                                                  self.smt_namespace.And(
    #                                                                      node.semantic_vars[Int] == 0 if not config.get().no_bind_free_semantic_vars else True,
    #                                                                      node.semantic_vars[Bool] == prod._get_rhs()
    #                                                                  )), f'semantic_constraint_bool_const_{prod.id}_{i_node}', True)
    #
    #     core = self.cores[0]
    #     core_size = len(core)
    #
    #     nodes_w_children = [n for n in self.nodes if n.children]
    #
    #     implies_rhs = self.smt_namespace.And([root.semantic_vars[Bool] for root in self.roots])
    #
    #     combo_lhss = []
    #     for node_combo in itertools.combinations(nodes_w_children, core_size):
    #         implies_lhs = []
    #         for node, core_atom in zip(node_combo, core):
    #             tmp = [node.var == self.spec.get_function_production_or_raise(core_atom.name).id]
    #             for child, core_atom_argument in zip(node.children, core_atom.arguments):
    #                 tmp.append(child.semantic_vars[Int] == core_atom_argument.number if core_atom_argument.type == clingo.SymbolType.Number else True)
    #             implies_lhs.append(self.smt_namespace.And(tmp))
    #
    #         implies_lhs = self.smt_namespace.And(*implies_lhs)
    #         combo_lhss.append(implies_lhs)
    #
    #     self.create_assertion(self.smt_namespace.And(self.smt_namespace.Or(combo_lhss), implies_rhs), f'core_{0}_restriction', True)

    @cached_property
    def max_children_except_aggregate(self) -> int:
        """Finds the maximum number of children in the productions"""
        max = 0
        for p in self.spec.get_function_productions():
            if p.name != 'aggregate':
                if len(p.rhs) > max:
                    max = len(p.rhs)
        return max

    @cached_property
    def max_children_with_aggregate(self) -> int:
        """Finds the maximum number of children in the productions"""
        max = 0
        for p in self.spec.get_function_productions():
            if len(p.rhs) > max:
                max = len(p.rhs)
        return max

    @staticmethod
    def _check_arg_types(pred, python_tys):
        if pred.num_args() < len(python_tys):
            msg = 'Predicate "{}" must have at least {} arugments. Only {} is found.'.format(
                pred.name, len(python_tys), pred.num_args())
            raise ValueError(msg)
        for index, (arg, python_ty) in enumerate(zip(pred.args, python_tys)):
            if not isinstance(arg, python_ty):
                msg = 'Argument {} of predicate {} has unexpected type.'.format(
                    index, pred.name)
                raise ValueError(msg)

    def _resolve_occurs_predicate(self, pred):
        self._check_arg_types(pred, [str, (int, float)])
        prod = self.spec.get_function_production_or_raise(pred.args[0])
        weight = pred.args[1]
        self.optimizer.mk_occurs(prod, weight)

    def _resolve_not_occurs_predicate(self, pred):
        if isinstance(pred.args[0], str):
            prod = self.spec.get_function_production_or_raise(pred.args[0])
        elif isinstance(pred.args[0], tuple):
            prod = self.spec.get_enum_production_or_raise(pred.args[0][0], pred.args[0][1])
        else:
            raise NotImplementedError()
        for stmt in self.statements:
            for node in stmt.nodes:
                if not node.bound:
                    self.create_assertion(node.var != prod.id)

    def _resolve_is_not_parent_predicate(self, pred):
        parent = self.spec.get_function_production_or_raise(pred.args[0])
        if isinstance(pred.args[1], str):
            child = self.spec.get_function_production_or_raise(pred.args[1])
        elif isinstance(pred.args[1], tuple):
            child = self.spec.get_enum_production(pred.args[1][0], pred.args[1][1])
        else:
            raise ValueError()

        child_pos = OrderedSet()
        # find positions that type-check between parent and child
        if len(pred.args) < 3:
            positions = range(0, len(parent.rhs))
        else:
            positions = pred.args[2]
        for x in positions:
            child_types = parent.rhs[x]
            if not isinstance(child_types, tuple):
                child_types = (child_types,)
            for child_type in child_types:
                if child.lhs == child_type or child_type.name == 'Any':
                    child_pos.append(x)
                    break

        for stmt in self.statements:
            for n in stmt.nodes:
                # not a leaf node
                if n.children and not n.bound:
                    ctr_children = []
                    for p in child_pos:
                        if p < len(n.children):
                            ctr_children.append(n.children[p].var == child.id)

                    self.create_assertion(self.smt_namespace.Implies(self.smt_namespace.Or(ctr_children), n.var != parent.id))

    def _resolve_is_parent_predicate(self, pred):
        self._check_arg_types(pred, [str, str, (int, float)])
        prod0 = self.spec.get_function_production_or_raise(pred.args[0])
        prod1 = self.spec.get_function_production_or_raise(pred.args[1])
        weight = pred.args[2]
        self.optimizer.mk_is_parent(prod0, prod1, weight)

    def _resolve_commutative_predicate(self, pred):
        if config.get().disable_commutative_predicate:
            return
        self._check_arg_types(pred, [str])
        prod = self.spec.get_function_production_or_raise(pred.args[0])
        self.commutative_prod_ids.append(prod.id)

    def _resolve_distinct_args_predicate(self, pred):
        if config.get().disable_distinct_args_predicate:
            return
        prod = self.spec.get_function_production_or_raise(pred.args[0])
        for stmt in self.statements:
            for node in stmt.nodes:
                self.create_distinct_args_recursive_constraints(prod, node)

    def create_distinct_args_recursive_constraints(self, prod, node: Node):
        ctr = []
        for child_A, child_B in itertools.combinations(node.children, 2):
            ctr_t = []
            for node_A, node_B in Node.collect_vars_pairwise(child_A, child_B):
                ctr_t.append(self.smt_namespace.Or(node_A != node_B, self.smt_namespace.And(node_A == 0, node_B == 0)))
            ctr.append(self.smt_namespace.Or(*ctr_t))
        self.create_assertion(self.smt_namespace.Implies(node.var == prod.id, self.smt_namespace.And(*ctr)))

    def resolve_predicates(self):
        try:
            for pred in self.spec.predicates():
                if pred.name == 'occurs':
                    self._resolve_occurs_predicate(pred)
                elif pred.name == 'is_parent':
                    self._resolve_is_parent_predicate(pred)
                elif pred.name == 'not_occurs':
                    self._resolve_not_occurs_predicate(pred)
                elif pred.name == 'is_not_parent':
                    self._resolve_is_not_parent_predicate(pred)
                elif pred.name == 'commutative':
                    self._resolve_commutative_predicate(pred)
                elif pred.name == 'distinct_args':
                    self._resolve_distinct_args_predicate(pred)
                else:
                    logger.warning('Predicate not handled: {}'.format(pred))
        except (KeyError, ValueError) as e:
            msg = 'Failed to resolve predicates. {}'.format(e)
            raise RuntimeError(msg) from None

    def blocking_template_compute(self) -> ExprRef:
        ctr = []
        for stmt in self.statements:
            for n in stmt.nodes:
                ctr.append(n.var != self.smt_namespace.Var(self.template_var_counter, self.smt_namespace.IntSort()))
                self.template_var_counter += 1
        return self.smt_namespace.Or(*ctr)

    def model_values(self, model, root=None):
        return [model[n.var] for stmt in self.statements for n in stmt.nodes]

    def block_model(self):
        self.solver.add(self.smt_namespace.substitute_vars(self.blocking_template, *self.model_values(self.model)))

    def update(self, info=None):
        # if info is not None and not isinstance(info, str):
        #     for core in info:
        #         ctr = []
        #         for constraint in core:
        #             ctr.append(self.variables[self.program2tree[constraint[0]].id - 1] != constraint[1].id)
        #         self.solver.add(Or(*ctr))
        # else:
        self.block_model()

    def build_program(self):
        built_stmts = []
        for stmt in self.statements:
            built_head = self.build_program_recursive(stmt.head)
            built_body = ApplyNode(self.and_production, [self.build_program_recursive(root) for root in stmt.body])
            built_stmts.append(ApplyNode(self.stmt_production, [built_head, built_body]))

        return ApplyNode(self.stmt_and_production, built_stmts)

    def build_program_recursive(self, root):
        if root is None:
            return

        prod = self.spec.get_production_or_raise(self.model[root.var].as_long())

        if prod.is_function() and prod.name != 'empty':
            return ApplyNode(prod, [self.build_program_recursive(root.children[i]) for i in range(len(prod.rhs))])
        elif prod.is_function() and prod.name == 'empty':
            return ApplyNode(prod, [])
        elif prod.is_enum():
            return AtomNode(prod)
        else:
            raise NotImplementedError()

    def next(self):
        runhelper.timer_start('z3.enum.time')
        res = self.solver.check()
        runhelper.timer_stop('z3.enum.time')

        if len(self.statements) == 0:
            return None

        if res == self.smt_namespace.unsat:
            self.unsat_core = self.solver.unsat_core()
            if not self.has_enumerated:
                logger.debug('Unsat core: %s', str(self.unsat_core))
            return None
        elif res == self.smt_namespace.unknown:
            logger.error('Z3 failed to produce an answer: %s', self.solver.reason_unknown())
            raise RuntimeError()

        self.has_enumerated = True
        self.model = self.solver.model()

        # print(self.model)

        if self.model is not None:
            runhelper.timer_start('program.build.time')
            program = self.build_program()
            runhelper.timer_stop('program.build.time')
            return program
        else:
            return None
