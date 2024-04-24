import itertools
import logging
from collections import namedtuple
from enum import Enum
from types import NoneType
from typing import Callable, TypeVar, NamedTuple, Any, Union

import z3
from ordered_set import OrderedSet
from z3 import Solver, ExprRef

import formhe.trinity.DSL
import runhelper
from formhe.asp.synthesis.ASPSpecGenerator import max_children_except_aggregate, max_children_with_aggregate
from formhe.trinity.DSL import TyrellSpec, ApplyNode, AtomNode
from formhe.trinity.DSL.type import Type
from formhe.utils import config

logger = logging.getLogger('formhe.asp.enumerator')

ExprType = TypeVar('ExprType')


class BindingType(Enum):
    NOT_BOUND = 0
    BOUND = 1
    SEMI_BOUND = 2


class AST:
    def __init__(self):
        self.head = None


class Node:
    def __init__(self, enumerator: 'NextGenEnumerator', id: int, depth: int, is_leaf: bool, n_child_indicator: int = None, parent: 'Node' = None):
        self.id = id
        self.depth = depth
        self.parent = parent
        self.children: list['Node'] = []
        self.is_leaf = is_leaf
        self.var = self._create_var(enumerator, n_child_indicator)
        # self.semantic_vars = self._create_semantic_vars(enumerator)
        self.bound: BindingType = BindingType.NOT_BOUND
        self.binding = None
        self.binding_relaxation_var = None
        self.spec = enumerator.spec

    def _create_var(self, enumerator: 'NextGenEnumerator', n_child_indicator: int = None) -> ExprRef:
        var = enumerator.create_variable(f'node_{self.id}', z3.Int)
        enumerator.create_assertion(var >= 0)
        enumerator.create_assertion(var < enumerator.spec.num_productions())
        ctr = []
        for function in enumerator.spec.get_function_productions():
            if self.is_leaf and function.name != 'empty':
                ctr.append(var != function.id)
            if n_child_indicator is not None and n_child_indicator < len(function.rhs):
                ctr.append(var != function.id)
        enumerator.create_assertion(z3.And(*ctr))

        return var

    def _create_semantic_vars(self, enumerator: 'NextGenEnumerator') -> dict[Type, ExprRef]:
        sem_vars = {
            enumerator.spec.get_type('Bool'): enumerator.create_variable(f'node_{self.id}_Bool', z3.Bool),
            enumerator.spec.get_type('Terminal'): enumerator.create_variable(f'node_{self.id}_Int', z3.Int),
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

    def bind(self, production_id: int, enumerator: 'NextGenEnumerator', binding_type: BindingType = BindingType.BOUND):
        if binding_type == BindingType.NOT_BOUND:
            return
        elif binding_type == BindingType.BOUND:
            self.bound = BindingType.BOUND
            self.binding = production_id
            enumerator.create_assertion(self.var == production_id, f'node_{self.id}_binding', True)
        elif binding_type == BindingType.SEMI_BOUND:
            self.bound = BindingType.SEMI_BOUND
            self.binding = production_id
            self.binding_relaxation_var = enumerator.create_variable(f'node_{self.id}_bind_counter', z3.Bool)
            # enumerator.create_assertion((self.var == production_id) != self.mutation_counter, f'node_{self.id}_binding', True)
            empty_prod = enumerator.spec.get_function_production('empty').id
            if self.parent is not None and self.parent.binding is not None and self.parent.binding != empty_prod:
                tmp = z3.And(self.var != production_id, z3.Not(z3.And(self.parent.var == empty_prod, self.var == empty_prod)))
            elif self.parent is not None and self.parent.binding == empty_prod:
                tmp = z3.And(self.var != production_id, self.parent.var == empty_prod)
            else:
                tmp = self.var != production_id
            enumerator.create_assertion(tmp == self.binding_relaxation_var, f'node_{self.id}_binding', True)

    def tree_repr(self, last=True, header=''):
        elbow = "└──"
        pipe = "│  "
        tee = "├──"
        blank = "   "
        res = header + (elbow if last else tee) + repr(self) + '\n'
        if self.children:
            for i, c in enumerate(self.children):
                res += c.tree_repr(header=header + (blank if last else pipe), last=i == len(self.children) - 1)
        return res

    def __repr__(self) -> str:
        # return f'Node({self.id}, vars={repr(self.collect_vars())}, semantic_variables=[{", ".join(map(lambda t: t.name, self.semantic_vars.keys()))}], bound={self.bound})'
        return f'Node({self.id}, var={self.var}, bound={self.bound.name}{", binding=" + str(self.spec.get_production(self.binding)) if self.bound else ""})'

    @staticmethod
    def collect_vars_pairwise(node_a, node_b):
        vars = [(node_a.var, node_b.var)]
        for child_a, child_b in zip(node_a.children, node_b.children):
            vars += Node.collect_vars_pairwise(child_a, child_b)
        return vars


class PresetStatement(NamedTuple):
    head: Any
    body: list[Any]

    def __repr__(self) -> str:
        r = ''
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

    def __init__(self, enumerator: 'NextGenEnumerator', tree_depth: int, preset_statement: PresetStatement, binding_type: BindingType = BindingType.BOUND, additional_body_roots: int = 0):
        self.id = enumerator.statement_counter
        enumerator.statement_counter += 1
        self.nodes: list[Node] = []
        self.head_nodes: list[Node] = []
        self.body_nodes: list[Node] = []
        self.binding_type = binding_type

        self.head = self.create_bound_tree(enumerator, preset_statement.head, tree_depth, max_children_with_aggregate(enumerator.spec), is_head=True, binding_type=binding_type)

        self.body = []
        for preset_atom in preset_statement.body:
            self.body.append(self.create_bound_tree(enumerator, preset_atom, tree_depth, max_children_except_aggregate(enumerator.spec), is_body=True, binding_type=binding_type))

        for i in range(additional_body_roots):
            self.body.append(self.create_bound_tree(enumerator, None, tree_depth, max_children_except_aggregate(enumerator.spec), is_body=True, binding_type=BindingType.SEMI_BOUND))

        self.create_head_output_constraints(enumerator, head_nullable=True)
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

    def finish_init(self, enumerator: 'NextGenEnumerator'):
        self.create_lexicographic_order_constraints(enumerator)

    def create_bound_tree(self, enumerator: 'NextGenEnumerator', ast_node: Union[formhe.trinity.DSL.Node, None], depth: int, max_children: int, current_depth: int = 1, is_body: bool = False, is_head: bool = False, parent: Node = None,
                          binding_type: BindingType = BindingType.BOUND) -> Node:
        node = Node(enumerator, enumerator.node_counter, 0, False, max_children, parent=parent)
        self.nodes.append(node)
        if is_body:
            self.body_nodes.append(node)
        if is_head:
            self.head_nodes.append(node)
        enumerator.node_counter += 1
        if ast_node is None:
            node.bind(0, enumerator, binding_type=binding_type)
            if current_depth < depth and not config.get().disable_mutation_node_expansion:
                for i in range(max_children):
                    node.children.append(self.create_bound_tree(enumerator, None, depth, max_children_except_aggregate(enumerator.spec), current_depth + 1, is_body, is_head, binding_type=BindingType.SEMI_BOUND, parent=node))
        elif isinstance(ast_node, ApplyNode):
            node.bind(ast_node.production.id, enumerator, binding_type=binding_type)
            for child in ast_node.children:
                node.children.append(self.create_bound_tree(enumerator, child, depth, max_children, current_depth + 1, is_body, is_head, binding_type=binding_type, parent=node))
            if current_depth < depth and len(ast_node.children) < max_children_except_aggregate(enumerator.spec) and not config.get().disable_mutation_node_expansion:
                for i in range(len(ast_node.children), max_children_except_aggregate(enumerator.spec)):
                    node.children.append(self.create_bound_tree(enumerator, None, depth, max_children, current_depth + 1, is_body, is_head, binding_type=BindingType.SEMI_BOUND, parent=node))
        elif isinstance(ast_node, AtomNode):
            node.bind(ast_node.production.id, enumerator, binding_type=binding_type)
            if current_depth < depth and not config.get().disable_mutation_node_expansion:
                for i in range(len(ast_node.children), max_children_except_aggregate(enumerator.spec)):
                    node.children.append(self.create_bound_tree(enumerator, None, depth, max_children, current_depth + 1, is_body, is_head, binding_type=BindingType.SEMI_BOUND, parent=node))

        return node

    def create_head_output_constraints(self, enumerator: 'NextGenEnumerator', head_nullable: bool):
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
        if self.head.bound == BindingType.NOT_BOUND:
            enumerator.create_assertion(z3.Or(*ctr))
        elif self.head.bound == BindingType.SEMI_BOUND:
            enumerator.create_assertion(z3.Or(self.head.var == self.head.binding, *ctr))

    def create_body_output_constraints(self, enumerator: 'NextGenEnumerator'):
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
            if root.bound == BindingType.NOT_BOUND:
                enumerator.create_assertion(z3.Or(*ctr))
            elif root.bound == BindingType.SEMI_BOUND:
                enumerator.create_assertion(z3.Or(root.var == root.binding, *ctr))

    def create_aggregate_restrictions_constraints(self, enumerator: 'NextGenEnumerator', root: Node):
        aggregate = enumerator.spec.get_function_production_or_raise('aggregate')
        constraint = []
        for var in root.collect_vars():
            constraint.append(var != aggregate.id)
        enumerator.create_assertion(z3.And(*constraint))

    def create_children_constraints(self, enumerator: 'NextGenEnumerator'):
        for node in self.nodes:
            for p in enumerator.spec.productions():
                if len(node.children) < len(p.rhs) and p.is_function() and p.name != "empty":
                    enumerator.create_assertion(node.var != p.id)
                else:
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
                        enumerator.create_assertion(z3.Implies(node.var == p.id, z3.Or(*ctr)))

    def create_head_empty_or_non_constant_constraints(self, enumerator: 'NextGenEnumerator'):
        if config.get().disable_head_empty_or_non_constant_constraint:
            return

        lhs_ctr = []
        for var in self.head.collect_vars():
            lhs_ctr.append(var == 0)
        lhs_ctr = z3.And(*lhs_ctr)
        rhs_ctr = []
        for production in enumerator.spec.productions():
            if (not production.is_constant and production.is_enum()) or (production.is_function() and production.name == 'interval'):
                # print(production)
                for var in self.head.collect_vars():
                    rhs_ctr.append(var == production.id)
        rhs_ctr = z3.Or(*rhs_ctr)
        if self.head.bound == BindingType.NOT_BOUND:
            enumerator.create_assertion(z3.Or(lhs_ctr, rhs_ctr), f'{self.id}_head_empty_or_non_const')
        elif self.head.bound == BindingType.SEMI_BOUND:
            bound_ctr = []
            for node in self.head.collect_nodes():
                bound_ctr.append(node.var == node.binding)
            bound_ctr = z3.And(*bound_ctr)
            enumerator.create_assertion(z3.Or(bound_ctr, lhs_ctr, rhs_ctr), f'{self.id}_head_empty_or_non_const')

    def create_no_dont_care_in_head_constraints(self, enumerator: 'NextGenEnumerator'):
        if config.get().disable_no_dont_care_in_head_constraint:
            return

        dont_care_prod = enumerator.spec.get_enum_production(enumerator.spec.get_type('Terminal'), '_')
        ctr = []
        if len(self.head.children) >= 4:
            for var in self.head.children[0].collect_vars() + self.head.children[1].collect_vars() + self.head.children[3].collect_vars():
                ctr.append(var != dont_care_prod.id)
            for var in self.head.children[2].collect_vars():
                ctr.append(z3.Or(self.head.var == enumerator.spec.get_function_production('aggregate').id, var != dont_care_prod.id))
        else:
            for var in self.head.collect_vars():
                ctr.append(var != dont_care_prod.id)
        enumerator.create_assertion(z3.And(*ctr), f'no_dont_care_in_head_{self.id}')

    def create_no_constant_sub_expressions_constraints(self, enumerator: 'NextGenEnumerator'):
        """For each subtree, either the tree is empty, or at least one non-const production is used"""
        subtrees = []
        for node in self.nodes:
            if node.children:
                subtrees.append(node.collect_nodes())

        for subtree in subtrees:
            ctr = []
            empty_ctr = []
            bound_ctr = []
            for node in subtree:
                empty_ctr.append(node.var == 0)
                bound_ctr.append(node.var == node.binding)
                for prod in enumerator.spec.productions():
                    if not prod.is_constant:
                        ctr.append(node.var == prod.id)
            if subtree[0].bound == BindingType.NOT_BOUND:
                enumerator.create_assertion(z3.Or(*ctr, z3.And(*empty_ctr)))
            elif subtree[0].bound == BindingType.SEMI_BOUND:
                enumerator.create_assertion(z3.Or(*ctr, z3.And(*empty_ctr), z3.And(*bound_ctr)))

    def create_no_constant_aggregate_constraints(self, enumerator: 'NextGenEnumerator'):
        agg_production = enumerator.spec.get_function_production('aggregate')

        ctr = []
        if len(self.head.children) >= 4:
            for var in self.head.children[1].collect_vars():
                for prod in enumerator.spec.productions():
                    if not prod.is_constant and prod.is_enum():
                        ctr.append(var == prod.id)
            enumerator.create_assertion(z3.Or(self.head.var != agg_production.id,
                                              z3.Or(*ctr)))

    def create_no_unsafe_vars_constraints(self, enumerator: 'NextGenEnumerator', free_vars: list):
        self.create_no_unsafe_vars_body_constraints(enumerator, free_vars)
        self.create_no_unsafe_vars_head_constraints(enumerator, free_vars)

    def create_no_unsafe_vars_body_constraints(self, enumerator: 'NextGenEnumerator', free_vars: list):
        if config.get().allow_unsafe_vars:
            return

        aux_vars = {v: enumerator.create_variable(f'var_used_body_{v}_stmt_{self.id}', z3.Bool) for v in free_vars}

        for var in free_vars:
            var_production = enumerator.spec.get_enum_production(enumerator.spec.get_type('Terminal'), var)
            ctr = []
            for root in self.body_nodes:
                if root.children:
                    sub_ctr = []
                    for predicate_name in enumerator.predicate_names:
                        production = enumerator.spec.get_function_production_or_raise(predicate_name)
                        sub_ctr.append(root.var == production.id)
                    sbt_ctr = z3.And(z3.Or(*sub_ctr),
                                     z3.Or(*[child.var == var_production.id for child in root.children]))
                    ctr.append(sbt_ctr)

            enumerator.create_assertion(aux_vars[var] == z3.Or(ctr))

            for node in self.body_nodes:
                enumerator.create_assertion(z3.Implies(z3.Not(aux_vars[var]), node.var != var_production.id))

            for node in self.head_nodes:
                enumerator.create_assertion(z3.Or(self.head.var == enumerator.spec.get_function_production('aggregate').id,
                                                  z3.Implies(z3.Not(aux_vars[var]), node.var != var_production.id)))

            if len(self.head.children) >= 4:
                for node in self.head.children[0].collect_nodes() + self.head.children[3].collect_nodes():
                    enumerator.create_assertion(z3.Or(self.head.var != enumerator.spec.get_function_production('aggregate').id,
                                                      z3.Implies(z3.Not(aux_vars[var]), node.var != var_production.id)))

    def create_no_unsafe_vars_head_constraints(self, enumerator: 'NextGenEnumerator', free_vars: list):
        if config.get().allow_unsafe_vars or len(self.head.children) < 4:
            return

        aux_vars = {v: enumerator.create_variable(f'var_used_head_{v}_stmt_{self.id}', z3.Bool) for v in free_vars}
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
                    sbt_ctr = z3.And(z3.Or(*sub_ctr),
                                     z3.Or(*[child.var == var_production.id for child in root.children]))
                    ctr.append(sbt_ctr)

            for root in self.head.children[2].collect_nodes():
                if root.children:
                    sub_ctr = []
                    for predicate_name in enumerator.predicate_names:
                        production = enumerator.spec.get_function_production_or_raise(predicate_name)
                        sub_ctr.append(root.var == production.id)
                    sbt_ctr = z3.And(self.head.var == agg_production.id,
                                     z3.Or(*sub_ctr),
                                     z3.Or(*[child.var == var_production.id for child in root.children]))
                    ctr.append(sbt_ctr)

            enumerator.create_assertion(aux_vars[var] == z3.Or(ctr))

            for node in self.head.children[1].collect_nodes():
                enumerator.create_assertion(z3.Or(self.head.var != agg_production.id,
                                                  z3.Implies(z3.Not(aux_vars[var]), node.var != var_production.id)))

    def create_lexicographic_order_constraints(self, enum: 'NextGenEnumerator', root: Node = None):
        if root is None:
            if len(self.body) > 1:
                fake_node = FakeNode(None, [r for r in self.body if r.bound == BindingType.NOT_BOUND], False)
                enum.create_assertion(self.create_lexicographic_order_constraints(enum, fake_node), f'lexicographic_body')
            elif len(self.body) > 0 and self.body[0].bound == BindingType.NOT_BOUND:
                enum.create_assertion(self.create_lexicographic_order_constraints(enum, self.body[0]), 'lexicographic_body')

            enum.create_assertion(self.create_lexicographic_order_constraints(enum, self.head), f'lexicographic_head')

        else:
            if root.children:
                root_is_commutative = []
                for p in enum.commutative_prod_ids:
                    root_is_commutative.append(root.var == p)
                if isinstance(root, FakeNode):
                    root_is_commutative.append(True)
                root_is_commutative = z3.Or(*root_is_commutative)

                lex_constraints = []
                for child_a, child_b in itertools.pairwise(root.children):
                    var_pairs = Node.collect_vars_pairwise(child_a, child_b)

                    lex_constraints.append(var_pairs[0][0] <= var_pairs[0][1])

                    for i, (var_a, var_b) in enumerate(var_pairs[1:]):
                        lex_constraints_tmp = []
                        for var_1, var_2 in var_pairs[:i]:
                            lex_constraints_tmp.append(var_1 == var_2)
                        lex_constraints.append(z3.Implies(z3.And(*lex_constraints_tmp), var_a <= var_b))

                semi_bound_guard = root.var != root.binding if root.bound == BindingType.SEMI_BOUND else True  # todo check me
                root_ctr = z3.Implies(z3.And(root_is_commutative, semi_bound_guard), z3.And(*lex_constraints))
                return z3.And(root_ctr, *[self.create_lexicographic_order_constraints(enum, child) for child in root.children])
            else:
                return True

    def collect_vars(self):
        return itertools.chain(self.head.collect_vars(), *[root.collect_vars() for root in self.body])

    def __repr__(self):
        res = f'Statement\n'
        res += self.head.tree_repr()
        for b in self.body:
            res += b.tree_repr()
        return res


FakeNode = namedtuple('FakeNode', ['var', 'children', 'bound'])


class BuildProgramException(Exception):
    pass


# FIXME: Currently this enumerator requires an "Empty" production to function properly
class NextGenEnumerator:

    def __init__(self, spec: TyrellSpec, depth=None, bound_statements=None, semi_bound_statements=None, strict_minimum_depth=True, free_predicates=None, force_generate_predicates=None, additional_body_roots=0):
        if bound_statements is None:
            bound_statements = []
        if semi_bound_statements is None:
            semi_bound_statements = []
        if free_predicates is None:
            free_predicates = []
        if force_generate_predicates is None:
            force_generate_predicates = []

        self.solver: Solver = self.init_solver()

        self.predicate_names = OrderedSet()

        self.var_counter = 0
        self.assertion_counter = 0
        self.template_var_counter = 0
        self.relaxation_template_var_counter = 0
        self.relaxation_block_template_var_counter = 0
        self.node_counter = 0
        self.statement_counter = 0

        self.spec = spec
        if depth <= 0:
            raise ValueError('Depth cannot be non-positive: {}'.format(depth))
        self.depth = depth
        self.var_counter = 0

        self.statements = [Statement(self, depth, preset_statement) for preset_statement in bound_statements] + \
                          [Statement(self, depth, preset_statement, binding_type=BindingType.SEMI_BOUND, additional_body_roots=additional_body_roots) for preset_statement in semi_bound_statements]

        debug_repr = ""
        for statement in self.statements:
            debug_repr += repr(statement)
        logger.debug('\n' + debug_repr)

        self.model = None
        # self.create_input_constraints()

        self.commutative_prod_ids = []

        self.resolve_predicates()

        for statement in self.statements:
            statement.finish_init(self)

        # if strict_minimum_depth:
        #     self.create_max_depth_used_constraints()
        # self.create_semantic_constraints()

        if not config.get().allow_not_generated_predicates:
            self.create_predicate_usage_constraints(free_predicates)
            self.create_predicate_force_generate_constraints(force_generate_predicates)

        self.blocking_template = self.blocking_template_compute()
        self.relaxation_template = self.relaxation_template_compute()
        self.relaxation_block_template = self.relaxation_block_template_compute()

        self.and_production = self.spec.get_function_production('and')
        self.stmt_and_production = self.spec.get_function_production('stmt_and')
        self.stmt_production = self.spec.get_function_production('stmt')
        self.has_enumerated = False

    def init_solver(self):
        # solver = z3.SolverFor("QF_NIA")
        solver = z3.Solver()

        solver.set('random_seed', config.get().seed)
        # solver.set('unsat_core', True)
        # solver.set('core.minimize', True)

        return solver

    def __del__(self):
        self.log()

    def log(self):
        logger.info('enum.smt.vars=%d', self.var_counter)
        logger.info('enum.smt.constraints=%d', self.assertion_counter)

    def create_variable(self, name: str, type: 'Callable[[str, ...], ExprType]', *args) -> ExprType:
        self.var_counter += 1
        return type(name, *args)

    def create_assertion(self, expr: z3.ExprRef, name: str = None, track: bool = False, debug_print: bool = False) -> None:
        # print(z3.simplify(expr))
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
    #         self.solver.add(z3.Or(*ctr))

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
                self.create_assertion(z3.Or(*ctr))

    def create_max_depth_used_constraints(self):
        ctr = []
        for smt in self.statements:
            for node in smt.nodes:
                if node.is_leaf and node.bound != BindingType.BOUND and node.depth == self.depth:
                    ctr.append(node.var != 0)
        self.create_assertion(z3.Or(*ctr), 'max_depth_used', False, False)

    def create_predicate_usage_constraints(self, predicates: list[str]):
        aggregate_prod = self.spec.get_function_production('aggregate')
        for predicate in predicates:
            predicate_prod = self.spec.get_function_production(predicate)

            lhs = []
            for stmt in self.statements:
                for node in stmt.body_nodes:
                    lhs.append(node.var == predicate_prod.id)
            lhs = z3.Or(*lhs)

            rhs = []
            for stmt in self.statements:
                if len(stmt.head.children) >= 4:
                    rhs.append(z3.Or(stmt.head.var == predicate_prod.id, z3.And(stmt.head.var == aggregate_prod.id, stmt.head.children[1].var == predicate_prod.id)))
                else:
                    rhs.append(stmt.head.var == predicate_prod.id)
            rhs = z3.Or(*rhs)

            self.create_assertion(z3.Implies(lhs, rhs), f'predicate_usage_{predicate}', False, False)

    def create_predicate_force_generate_constraints(self, predicates: list[str]):
        aggregate_prod = self.spec.get_function_production('aggregate')
        for predicate in predicates:
            predicate_prod = self.spec.get_function_production(predicate)

            rhs = []
            for stmt in self.statements:
                if len(stmt.head.children) >= 4:
                    rhs.append(z3.Or(stmt.head.var == predicate_prod.id, z3.And(stmt.head.var == aggregate_prod.id, stmt.head.children[1].var == predicate_prod.id)))
                else:
                    rhs.append(stmt.head.var == predicate_prod.id)
            rhs = z3.Or(*rhs)

            self.create_assertion(rhs, f'predicate_force_{predicate}', False, False)

    def set_mutation_count(self, i: int):
        mutation_vars = []
        for stmt in self.statements:
            for node in stmt.nodes:
                if node.bound == BindingType.SEMI_BOUND:
                    mutation_vars.append(node.binding_relaxation_var)

        # self.create_assertion(z3.Sum(*mutation_vars) == i, "mutation_count_bound")
        self.create_assertion(z3.PbEq([(var, 1) for var in mutation_vars], i), "mutation_count_bound", True)

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
    #         free_vars_semantic_vars[var][Bool] = self.create_variable(f'var_{var}_Bool', z3.Bool)
    #         free_vars_semantic_vars[var][Terminal] = self.create_variable(f'var_{var}_Int', z3.Int)
    #
    #     for i_node, node in enumerate(self.nodes):
    #         self.create_assertion(z3.Implies(node.var == self.spec.get_function_production_or_raise('empty').id,
    #                                                          z3.And(
    #                                                              node.semantic_vars[Int] == 0,
    #                                                              z3.Not(node.semantic_vars[Bool])
    #                                                          )), f'semantic_constraint_empty_{i_node}', True)
    #
    #         if node.children:
    #             self.create_assertion(z3.Implies(node.var == self.spec.get_function_production_or_raise('and').id,
    #                                                              z3.And(
    #                                                                  node.semantic_vars[Int] == 0,
    #                                                                  node.semantic_vars[Bool] == z3.And([c.semantic_vars[Bool] for c in node.children])
    #                                                              )), f'semantic_constraint_and_{i_node}', True)
    #
    #             self.create_assertion(z3.Implies(node.var == self.spec.get_function_production_or_raise('add').id,
    #                                                              z3.And(
    #                                                                  node.semantic_vars[Int] == node.children[0].semantic_vars[Int] + node.children[1].semantic_vars[Int],
    #                                                                  z3.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                              )), f'semantic_constraint_add_{i_node}', True)
    #
    #             self.create_assertion(z3.Implies(node.var == self.spec.get_function_production_or_raise('sub').id,
    #                                                              z3.And(
    #                                                                  node.semantic_vars[Int] == (node.children[0].semantic_vars[Int] - node.children[1].semantic_vars[Int]),
    #                                                                  z3.Not(node.semantic_vars[Bool])
    #                                                              )), f'semantic_constraint_sub_{i_node}', True)
    #
    #             self.create_assertion(z3.Implies(node.var == self.spec.get_function_production_or_raise('mul').id,
    #                                                              z3.And(
    #                                                                  node.semantic_vars[Int] == node.children[0].semantic_vars[Int] * node.children[1].semantic_vars[Int],
    #                                                                  z3.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                              )), f'semantic_constraint_mul_{i_node}', True)
    #
    #             self.create_assertion(z3.Implies(node.var == self.spec.get_function_production_or_raise('div').id,
    #                                                              z3.And(
    #                                                                  node.semantic_vars[Int] == node.children[0].semantic_vars[Int] / node.children[1].semantic_vars[Int],
    #                                                                  z3.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                              )), f'semantic_constraint_div_{i_node}', True)
    #
    #             self.create_assertion(z3.Implies(node.var == self.spec.get_function_production_or_raise('abs').id,
    #                                                              z3.And(
    #                                                                  node.semantic_vars[Int] == z3.Abs(node.children[0].semantic_vars[Int]),
    #                                                                  z3.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                              )), f'semantic_constraint_abs_{i_node}', True)
    #
    #             self.create_assertion(z3.Implies(node.var == self.spec.get_function_production_or_raise('eq').id,
    #                                                              z3.And(
    #                                                                  node.semantic_vars[Bool] == (node.children[0].semantic_vars[Int] == node.children[1].semantic_vars[Int]),
    #                                                                  node.semantic_vars[Bool] == (node.children[0].semantic_vars[Bool] == node.children[1].semantic_vars[Bool])
    #                                                              )), f'semantic_constraint_eq_{i_node}', True)
    #
    #             self.create_assertion(z3.Implies(node.var == self.spec.get_function_production_or_raise('neq').id,
    #                                                              z3.Or(
    #                                                                  node.semantic_vars[Bool] == (node.children[0].semantic_vars[Int] != node.children[1].semantic_vars[Int]),
    #                                                                  node.semantic_vars[Bool] == (node.children[0].semantic_vars[Bool] != node.children[1].semantic_vars[Bool])
    #                                                              )), f'semantic_constraint_neq_{i_node}', True)
    #
    #         for prod in self.spec.get_productions_with_lhs(Int):
    #             if prod.is_enum() and prod.is_constant:
    #                 self.create_assertion(z3.Implies(node.var == prod.id,
    #                                                                  z3.And(
    #                                                                      node.semantic_vars[Int] == prod._get_rhs(),
    #                                                                      z3.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
    #                                                                  )), f'semantic_constraint_int_const_{prod.id}_{i_node}', True)
    #         for var in self.free_vars:
    #             prod = self.spec.get_enum_production(Int, var)
    #             self.create_assertion(z3.Implies(node.var == prod.id,
    #                                                              z3.And(
    #                                                                  node.semantic_vars[Int] == free_vars_semantic_vars[var][Int],
    #                                                                  node.semantic_vars[Bool] == free_vars_semantic_vars[var][Bool]
    #                                                              )), f'semantic_constraint_var_{var}_{i_node}', True)
    #
    #         for prod in self.spec.get_productions_with_lhs(Bool):
    #             if prod.is_enum() and prod.is_constant:
    #                 self.create_assertion(z3.Implies(node.var == prod.id,
    #                                                                  z3.And(
    #                                                                      node.semantic_vars[Int] == 0 if not config.get().no_bind_free_semantic_vars else True,
    #                                                                      node.semantic_vars[Bool] == prod._get_rhs()
    #                                                                  )), f'semantic_constraint_bool_const_{prod.id}_{i_node}', True)
    #
    #     core = self.cores[0]
    #     core_size = len(core)
    #
    #     nodes_w_children = [n for n in self.nodes if n.children]
    #
    #     implies_rhs = z3.And([root.semantic_vars[Bool] for root in self.roots])
    #
    #     combo_lhss = []
    #     for node_combo in itertools.combinations(nodes_w_children, core_size):
    #         implies_lhs = []
    #         for node, core_atom in zip(node_combo, core):
    #             tmp = [node.var == self.spec.get_function_production_or_raise(core_atom.name).id]
    #             for child, core_atom_argument in zip(node.children, core_atom.arguments):
    #                 tmp.append(child.semantic_vars[Int] == core_atom_argument.number if core_atom_argument.type == clingo.SymbolType.Number else True)
    #             implies_lhs.append(z3.And(tmp))
    #
    #         implies_lhs = z3.And(*implies_lhs)
    #         combo_lhss.append(implies_lhs)
    #
    #     self.create_assertion(z3.And(z3.Or(combo_lhss), implies_rhs), f'core_{0}_restriction', True)

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
                if node.bound == BindingType.NOT_BOUND:
                    self.create_assertion(node.var != prod.id)
                elif node.bound == BindingType.SEMI_BOUND:  # fixme hack
                    self.create_assertion(z3.Or(node.var == node.binding, node.var != prod.id))

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
                if n.children and (n.bound != BindingType.BOUND):
                    ctr_children = []
                    for p in child_pos:
                        if p < len(n.children):
                            ctr_children.append(n.children[p].var == child.id)

                    if n.bound == BindingType.NOT_BOUND:
                        self.create_assertion(z3.Implies(z3.Or(ctr_children), n.var != parent.id))
                    elif n.bound == BindingType.SEMI_BOUND:
                        self.create_assertion(z3.Or(n.var == n.binding, z3.Implies(z3.Or(ctr_children), n.var != parent.id)))

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

    def _resolve_is_predicate_predicate(self, pred):
        self.predicate_names.append(pred.args[0])

    def create_distinct_args_recursive_constraints(self, prod, node: Node):
        ctr = []
        for child_A, child_B in itertools.combinations(node.children, 2):
            ctr_t = []
            for node_A, node_B in Node.collect_vars_pairwise(child_A, child_B):
                ctr_t.append(z3.Or(node_A != node_B, z3.And(node_A == 0, node_B == 0)))
            ctr.append(z3.Or(*ctr_t))
        self.create_assertion(z3.Implies(node.var == prod.id, z3.And(*ctr)))

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
                elif pred.name == 'is_predicate':
                    self._resolve_is_predicate_predicate(pred)
                else:
                    logger.warning('Predicate not handled: {}'.format(pred))
        except (KeyError, ValueError) as e:
            msg = 'Failed to resolve predicates. {}'.format(e)
            raise RuntimeError(msg) from None

    def blocking_template_compute(self) -> ExprRef:
        ctr = []
        for stmt in self.statements:
            for n in stmt.nodes:
                ctr.append(n.var != z3.Var(self.template_var_counter, z3.IntSort()))
                self.template_var_counter += 1
        return z3.Or(*ctr)

    def model_values(self, model, root=None):
        return [model[n.var] for stmt in self.statements for n in stmt.nodes]

    def model_values_tree(self, model: z3.ModelRef, root=None):
        if root is None:
            return [[self.model_values_tree(model, stmt.head), *(self.model_values_tree(model, body) for body in stmt.body)] for stmt in self.statements]
        if not root.children:
            v = int(model[root.var].as_long())
            prod = self.spec.get_production(v)
            if prod.is_function():
                text = prod.name
            elif prod.is_enum():
                text = prod.rhs[0]
            if root.bound and root.binding == v:
                text = "." + text
            return text if v != 0 else 'empty'
        else:
            v = int(model[root.var].as_long())
            prod = self.spec.get_production(v)
            if prod.is_function():
                text = prod.name
            elif prod.is_enum():
                text = prod.rhs[0]
            if root.bound and root.binding == v:
                text = "." + text
            return {text: [self.model_values_tree(model, c) for c in root.children]} if v != 0 else 'empty'

    def block_model(self, model=None):
        if model is None:
            model = self.model
        self.solver.add(z3.substitute_vars(self.blocking_template, *self.model_values(model)))

    def relaxation_vars(self):
        return [node.binding_relaxation_var for stmt in self.statements for node in stmt.nodes if node.bound != BindingType.NOT_BOUND]

    def model_relaxation_values(self, model, root=None):
        return [model[n.binding_relaxation_var] for stmt in self.statements for n in stmt.nodes]

    def relaxation_template_compute(self):
        ctr = []
        for stmt in self.statements:
            for n in stmt.nodes:
                if n.bound != BindingType.NOT_BOUND:
                    ctr.append(n.binding_relaxation_var == z3.Var(self.relaxation_template_var_counter, z3.BoolSort()))
                    self.relaxation_template_var_counter += 1
        return z3.And(*ctr)

    def set_relaxation_vars(self, perm: list[bool]):
        self.solver.add(z3.substitute_vars(self.relaxation_template, *[z3.BoolVal(p) for p in perm]))

    def relaxation_block_template_compute(self):
        ctr = []
        for stmt in self.statements:
            for n in stmt.nodes:
                if n.bound != BindingType.NOT_BOUND:
                    ctr.append(n.binding_relaxation_var != z3.Var(self.relaxation_block_template_var_counter, z3.BoolSort()))
                    self.relaxation_block_template_var_counter += 1
        return z3.Or(*ctr)

    def block_relaxation(self, perm: list[bool]):
        self.solver.add(z3.substitute_vars(self.relaxation_block_template, *[z3.BoolVal(p) for p in perm]))

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
            if stmt.binding_type == BindingType.BOUND:
                continue
            built_head = self.build_program_recursive(stmt.head)
            body = [self.build_program_recursive(root) for root in stmt.body]
            if built_head.production.id == 0 and all(map(lambda b: b.production.id == 0, body)) and stmt.head.binding == 0 and any(map(lambda b: b.binding == 0, stmt.body)):
                continue
            built_body = ApplyNode(self.and_production, body)
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

    def next(self, *assumptions, timeout=None):
        runhelper.timer_start('enum.fail.time')
        runhelper.timer_start('enum.z3.time')
        res = self.solver.check(*assumptions)
        runhelper.timer_stop('enum.z3.time')

        if res == z3.unsat:
            # self.unsat_core = self.solver.unsat_core()
            # if not self.has_enumerated:
            #     logger.debug('Unsat core: %s', str(self.unsat_core))
            return None
        elif res == z3.unknown:
            logger.error('Z3 failed to produce an answer: %s', self.solver.reason_unknown())
            return None
            # raise RuntimeError()

        self.has_enumerated = True
        self.model = self.solver.model()

        if self.model is not None:
            runhelper.timer_start('program.build.time')
            try:
                program = self.build_program()
            except Exception as e:
                runhelper.timer_stop('program.build.time')
                runhelper.timer_stop('enum.fail.time')
                logger.info("Failed to build program", exc_info=True)
                runhelper.tag_increment('enum.fail.programs')
                self.block_model()
                raise BuildProgramException(e)
            runhelper.timer_stop('program.build.time')
            self.block_model()
            return program
        else:
            logger.error('Z3 formula was SAT but there is no model')
            raise RuntimeError()
