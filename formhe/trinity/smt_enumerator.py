import itertools
import logging
from abc import ABC
from collections import namedtuple, defaultdict
from functools import cached_property
from typing import Callable, TypeVar, Union

import cvc5.pythonic
import z3
from ordered_set import OrderedSet

import formhe.trinity.DSL
from formhe.trinity.DSL import TyrellSpec, ApplyNode, AtomNode
from formhe.trinity.DSL.type import Type, EnumType
from formhe.utils import config, perf

logger = logging.getLogger('formhe.asp.enumerator')

ExprType = TypeVar('ExprType')

Solver = Union[z3.Solver, cvc5.pythonic.Solver]
ExprRef = Union[z3.ExprRef, cvc5.pythonic.ExprRef]


class AST:
    def __init__(self):
        self.head = None


class Node:
    def __init__(self, enumerator: 'SmtEnumerator', id: int, depth: int, is_leaf: bool):
        self.id = id
        self.depth = depth
        self.children: list['Node'] = []
        self.is_leaf = is_leaf
        self.var = self._create_var(enumerator)
        self.semantic_vars = self._create_semantic_vars(enumerator)
        self.bound = False

    def _create_var(self, enumerator: 'SmtEnumerator') -> ExprRef:
        var = enumerator.create_variable(f'node_{self.id}', enumerator.smt_namespace.Int)
        enumerator.solver.add(var >= 0)
        enumerator.solver.add(var < enumerator.spec.num_productions())
        if self.is_leaf:  # if node is a leaf then functions are not allowed
            ctr = []
            for p in enumerator.spec.get_function_productions():
                if p.name != 'empty':  # except for the empty function
                    ctr.append(var != p.id)
            enumerator.solver.add(enumerator.smt_namespace.And(*ctr))

        return var

    def _create_semantic_vars(self, enumerator: 'SmtEnumerator') -> dict[Type, ExprRef]:
        sem_vars = {
            enumerator.spec.get_type('Bool'): enumerator.create_variable(f'node_{self.id}_Bool', enumerator.smt_namespace.Bool),
            enumerator.spec.get_type('Int'): enumerator.create_variable(f'node_{self.id}_Int', enumerator.smt_namespace.Int),
        }

        return sem_vars

    def collect_vars(self) -> list[ExprRef]:
        vars = [self.var]
        for child in self.children:
            vars += child.collect_vars()
        return vars

    def bind(self, production_id: int, enumerator: 'SmtEnumerator'):
        self.bound = True
        print(self.var == production_id)
        enumerator.create_assertion(self.var == production_id, f'node_{self.id}_binding', True)

    def __repr__(self) -> str:
        return f'Node({self.id}, semantic_variables=[{", ".join(map(lambda t: t.name, self.semantic_vars.keys()))}])'


FakeNode = namedtuple('FakeNode', ['var', 'children', 'bound'])


# FIXME: Currently this enumerator requires an "Empty" production to function properly
class SmtEnumerator(ABC):
    # productions that are leaf
    leaf_productions = []

    # map from internal k-tree to nodes of program
    program2tree = {}

    def __init__(self, spec: TyrellSpec, depth=None, n_roots=1, predicates_names=None, cores=None, free_vars=None, preset_atoms=None):
        if predicates_names is None:
            predicates_names = []
        if preset_atoms is None:
            preset_atoms = []

        self.solver: Solver = self.init_solver()

        self.predicate_names = predicates_names
        self.cores = cores
        self.free_vars = free_vars

        self.var_counter = 0
        self.assertion_counter = 0
        self.template_var_counter = 0
        self.node_counter = 0

        self.leaf_productions = []
        self.nodes: list[Node] = []
        self.program2tree = {}
        self.spec = spec
        if depth <= 0:
            raise ValueError('Depth cannot be non-positive: {}'.format(depth))
        self.depth = depth
        self.var_counter = 0
        self.roots = []
        for i in range(n_roots):
            self.roots.append(self.build_ktree())
        if not config.get().no_semantic_constraints:
            for preset_atom in preset_atoms:
                self.roots.append(self.build_bound_ktree(preset_atom))
        # for node in self.nodes:
        #     print('   ' * (node.depth - 1) + str(node))
        self.model = None
        self.create_output_constraints()
        self.create_input_constraints()
        self.create_children_constraints()
        if not config.get().allow_constant_expressions:
            self.create_no_constant_sub_expressions_constraints(self.solver)

        self.commutative_prod_ids = []

        self.resolve_predicates()

        self.create_lexicographic_order_constraints()
        self.create_semantic_constraints()

        self.blocking_template = self.blocking_template_compute()

        self.and_production = self.spec.get_function_production('and')
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
            print(expr)
        self.assertion_counter += 1
        if not config.get().no_enumerator_debug and track and name is not None:
            self.solver.assert_and_track(expr, name)
        else:
            self.solver.add(expr)

    def create_output_constraints(self):
        '''The output production matches the output type'''
        output_tys = OrderedSet()
        if isinstance(self.spec.output, tuple):
            for ty in self.spec.output:
                output_tys.append(ty)
        else:
            output_tys.append(self.spec.output)
        for root in self.roots:
            ctr = []
            for ty in output_tys:
                for p in self.spec.get_productions_with_lhs(ty):
                    ctr.append(root.var == p.id)
            self.solver.add(self.smt_namespace.Or(*ctr))

    def create_input_constraints(self):
        '''Each input will appear at least once in the program'''
        input_productions = self.spec.get_param_productions()
        for x in range(0, len(input_productions)):
            ctr = []
            for node in self.nodes:
                ctr.append(node.var == input_productions[x].id)
            self.solver.add(self.smt_namespace.Or(*ctr))

    def create_children_constraints(self):
        for node in self.nodes:
            if node.children:
                for p in self.spec.productions():
                    assert len(node.children) > 0
                    for child_i, child in enumerate(node.children):
                        ctr = []
                        child_type = 'Empty'
                        if p.is_function() and child_i < len(p.rhs):
                            child_type = str(p.rhs[child_i])
                        if child_type != 'Any':
                            for t in self.spec.get_productions_with_lhs(child_type):
                                ctr.append(child.var == t.id)
                        else:
                            for t in self.spec.productions():
                                if t.lhs.name != 'Empty':
                                    ctr.append(child.var == t.id)
                        self.solver.add(self.smt_namespace.Implies(node.var == p.id, self.smt_namespace.Or(*ctr)))

    def create_no_constant_sub_expressions_constraints(self, solver: Solver):
        subtrees = []
        for node in self.nodes:
            if node.children:
                subtrees.append(node.collect_vars())

        for subtree in subtrees:
            ctr = []
            ctr_2 = []
            for node in subtree:
                ctr_2.append(node == 0)
                for prod in self.spec.productions():
                    if not prod.is_constant:
                        ctr.append(node == prod.id)
            solver.add(self.smt_namespace.Or(*ctr, self.smt_namespace.And(*ctr_2)))

    def create_no_unsafe_vars_constraints(self, free_vars: list):
        if config.get().allow_unsafe_vars:
            return

        aux_vars = {v: self.create_variable(f'var_used_{v}', self.smt_namespace.Bool) for v in free_vars}

        root_sbt_var = self.smt_namespace.Var(0, self.smt_namespace.IntSort())
        child_sbt_var_1 = self.smt_namespace.Var(1, self.smt_namespace.IntSort())
        child_sbt_var_2 = self.smt_namespace.Var(2, self.smt_namespace.IntSort())
        free_var_sbt_var = self.smt_namespace.Var(3, self.smt_namespace.IntSort())
        ctr = []
        for predicate_name in self.predicate_names:
            production = self.spec.get_function_production_or_raise(predicate_name)
            ctr.append(root_sbt_var == production.id)
        sbt_ctr = self.smt_namespace.And(self.smt_namespace.Or(*ctr), self.smt_namespace.Or(child_sbt_var_1 == free_var_sbt_var, child_sbt_var_2 == free_var_sbt_var))

        for var in free_vars:
            var_production = self.spec.get_enum_production(self.spec.get_type('Int'), var)
            ctr = []
            for root in self.nodes:
                if root.children:
                    tmp = self.smt_namespace.substitute_vars(sbt_ctr,
                                                             root.var,
                                                             root.children[0].var,
                                                             root.children[1].var,
                                                             self.smt_namespace.IntVal(var_production.id))

                    ctr.append(tmp)

            self.solver.add(aux_vars[var] == self.smt_namespace.Or(ctr))

            for node in self.nodes:
                self.solver.add(self.smt_namespace.Implies(self.smt_namespace.Not(aux_vars[var]), node.var != var_production.id))

    def create_force_var_usage_constraints(self, unsafe_vars: list):
        if config.get().allow_unsafe_vars:
            return

        for var in unsafe_vars:
            var_production = self.spec.get_enum_production(self.spec.get_type('Int'), var)
            ctr = []
            for node in self.nodes:
                ctr.append(node.var == var_production.id)
            self.solver.add(self.smt_namespace.Or(*ctr))

    def create_lexicographic_order_constraints(self, root: Node = None):
        if root is None:
            if len(self.roots) > 1:
                for constraint_i, (A, B) in enumerate(itertools.pairwise([r for r in self.roots if not r.bound])):
                    fake_node = FakeNode(self.commutative_prod_ids[0], [A, B], False)
                    self.create_assertion(self.create_lexicographic_order_constraints(fake_node), f'lexicographic_{constraint_i}', True)
                    self.solver.add()
            elif not self.roots[0].bound:
                self.create_assertion(self.create_lexicographic_order_constraints(self.roots[0]), 'lexicographic', True)

        else:
            if root.children:
                root_is_commutative = []
                for p in self.commutative_prod_ids:
                    root_is_commutative.append(root.var == p)
                root_is_commutative = self.smt_namespace.Or(*root_is_commutative)

                subtree_vars_A = root.children[0].collect_vars()
                subtree_vars_B = root.children[1].collect_vars()

                lex_constraints = [subtree_vars_A[0] <= subtree_vars_B[0]]
                for i in range(1, len(subtree_vars_A)):
                    lex_constraints_tmp = []
                    for j in range(i):
                        lex_constraints_tmp.append(subtree_vars_A[j] == subtree_vars_B[j])
                    lex_constraints.append(self.smt_namespace.Implies(self.smt_namespace.And(*lex_constraints_tmp), subtree_vars_A[i] <= subtree_vars_B[i]))

                root_ctr = self.smt_namespace.Implies(root_is_commutative, self.smt_namespace.And(*lex_constraints))
                return self.smt_namespace.And(root_ctr, self.create_lexicographic_order_constraints(root.children[0]), self.create_lexicographic_order_constraints(root.children[1]))
            else:
                return True

    def create_semantic_constraints(self):
        if self.cores is None or config.get().no_semantic_constraints:
            return

        Bool: EnumType = self.spec.get_type('Bool')
        Int: EnumType = self.spec.get_type('Int')

        free_vars_semantic_vars = defaultdict(dict)

        for var in self.free_vars:
            free_vars_semantic_vars[var][Bool] = self.create_variable(f'var_{var}_Bool', self.smt_namespace.Bool)
            free_vars_semantic_vars[var][Int] = self.create_variable(f'var_{var}_Int', self.smt_namespace.Int)

        for i_node, node in enumerate(self.nodes):
            self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('empty').id,
                                                             self.smt_namespace.And(
                                                                 node.semantic_vars[Int] == 0,
                                                                 self.smt_namespace.Not(node.semantic_vars[Bool])
                                                             )), f'semantic_constraint_empty_{i_node}', True)

            if node.children:
                self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('and').id,
                                                                 self.smt_namespace.And(
                                                                     node.semantic_vars[Int] == 0,
                                                                     node.semantic_vars[Bool] == self.smt_namespace.And([c.semantic_vars[Bool] for c in node.children])
                                                                 )), f'semantic_constraint_and_{i_node}', True)

                self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('add').id,
                                                                 self.smt_namespace.And(
                                                                     node.semantic_vars[Int] == node.children[0].semantic_vars[Int] + node.children[1].semantic_vars[Int],
                                                                     self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
                                                                 )), f'semantic_constraint_add_{i_node}', True)

                self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('sub').id,
                                                                 self.smt_namespace.And(
                                                                     node.semantic_vars[Int] == (node.children[0].semantic_vars[Int] - node.children[1].semantic_vars[Int]),
                                                                     self.smt_namespace.Not(node.semantic_vars[Bool])
                                                                 )), f'semantic_constraint_sub_{i_node}', True)

                self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('mul').id,
                                                                 self.smt_namespace.And(
                                                                     node.semantic_vars[Int] == node.children[0].semantic_vars[Int] * node.children[1].semantic_vars[Int],
                                                                     self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
                                                                 )), f'semantic_constraint_mul_{i_node}', True)

                self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('div').id,
                                                                 self.smt_namespace.And(
                                                                     node.semantic_vars[Int] == node.children[0].semantic_vars[Int] / node.children[1].semantic_vars[Int],
                                                                     self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
                                                                 )), f'semantic_constraint_div_{i_node}', True)

                self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('abs').id,
                                                                 self.smt_namespace.And(
                                                                     node.semantic_vars[Int] == self.smt_namespace.Abs(node.children[0].semantic_vars[Int]),
                                                                     self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
                                                                 )), f'semantic_constraint_abs_{i_node}', True)

                self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('eq').id,
                                                                 self.smt_namespace.And(
                                                                     node.semantic_vars[Bool] == (node.children[0].semantic_vars[Int] == node.children[1].semantic_vars[Int]),
                                                                     node.semantic_vars[Bool] == (node.children[0].semantic_vars[Bool] == node.children[1].semantic_vars[Bool])
                                                                 )), f'semantic_constraint_eq_{i_node}', True)

                self.create_assertion(self.smt_namespace.Implies(node.var == self.spec.get_function_production_or_raise('neq').id,
                                                                 self.smt_namespace.Or(
                                                                     node.semantic_vars[Bool] == (node.children[0].semantic_vars[Int] != node.children[1].semantic_vars[Int]),
                                                                     node.semantic_vars[Bool] == (node.children[0].semantic_vars[Bool] != node.children[1].semantic_vars[Bool])
                                                                 )), f'semantic_constraint_neq_{i_node}', True)

            for prod in self.spec.get_productions_with_lhs(Int):
                if prod.is_enum() and prod.is_constant:
                    self.create_assertion(self.smt_namespace.Implies(node.var == prod.id,
                                                                     self.smt_namespace.And(
                                                                         node.semantic_vars[Int] == prod._get_rhs(),
                                                                         self.smt_namespace.Not(node.semantic_vars[Bool]) if not config.get().no_bind_free_semantic_vars else True
                                                                     )), f'semantic_constraint_int_const_{prod.id}_{i_node}', True)
            for var in self.free_vars:
                prod = self.spec.get_enum_production(Int, var)
                self.create_assertion(self.smt_namespace.Implies(node.var == prod.id,
                                                                 self.smt_namespace.And(
                                                                     node.semantic_vars[Int] == free_vars_semantic_vars[var][Int],
                                                                     node.semantic_vars[Bool] == free_vars_semantic_vars[var][Bool]
                                                                 )), f'semantic_constraint_var_{var}_{i_node}', True)

            for prod in self.spec.get_productions_with_lhs(Bool):
                if prod.is_enum() and prod.is_constant:
                    self.create_assertion(self.smt_namespace.Implies(node.var == prod.id,
                                                                     self.smt_namespace.And(
                                                                         node.semantic_vars[Int] == 0 if not config.get().no_bind_free_semantic_vars else True,
                                                                         node.semantic_vars[Bool] == prod._get_rhs()
                                                                     )), f'semantic_constraint_bool_const_{prod.id}_{i_node}', True)

        core = self.cores[0]
        print(core)
        core_size = len(core)

        nodes_w_children = [n for n in self.nodes if n.children]

        implies_rhs = self.smt_namespace.And([root.semantic_vars[Bool] for root in self.roots])

        combo_lhss = []
        for node_combo in itertools.combinations(nodes_w_children, core_size):
            implies_lhs = []
            for node, core_atom in zip(node_combo, core):
                tmp = self.smt_namespace.And(node.var == self.spec.get_function_production_or_raise(core_atom.name).id,
                                             node.children[0].semantic_vars[Int] == core_atom.arguments[0].number,
                                             node.children[1].semantic_vars[Int] == core_atom.arguments[1].number)
                implies_lhs.append(tmp)

            implies_lhs = self.smt_namespace.And(*implies_lhs)
            combo_lhss.append(implies_lhs)

        self.create_assertion(self.smt_namespace.And(self.smt_namespace.Or(combo_lhss), implies_rhs), f'core_{0}_restriction', True)

    @cached_property
    def max_children(self) -> int:
        """Finds the maximum number of children in the productions"""
        max = 0
        for p in self.spec.get_function_productions():
            if len(p.rhs) > max:
                max = len(p.rhs)
        return max

    def build_ktree(self, current_depth: int = 1) -> Node:
        """Builds a K-tree that will contain the program"""
        node = Node(self, self.node_counter, current_depth, current_depth == self.depth)
        self.nodes.append(node)
        self.node_counter += 1
        if current_depth < self.depth:
            for i in range(self.max_children):
                node.children.append(self.build_ktree(current_depth + 1))
        return node

    def build_bound_ktree(self, ast_node: trinity.DSL.Node, current_depth: int = 1) -> Node:
        node = Node(self, self.node_counter, current_depth, current_depth == self.depth)
        self.nodes.append(node)
        self.node_counter += 1
        if ast_node is None:
            node.bind(0, self)
            if current_depth < self.depth:
                for i in range(self.max_children):
                    node.children.append(self.build_bound_ktree(None, current_depth + 1))
        elif isinstance(ast_node, ApplyNode):
            node.bind(ast_node.production.id, self)
            if current_depth < self.depth or ast_node.children:
                for i in range(self.max_children):
                    if i < len(ast_node.children):
                        node.children.append(self.build_bound_ktree(ast_node.children[i], current_depth + 1))
                    else:
                        node.children.append(self.build_bound_ktree(None, current_depth + 1))
        elif isinstance(ast_node, AtomNode):
            node.bind(ast_node.production.id, self)
            if current_depth < self.depth:
                for i in range(self.max_children):
                    node.children.append(self.build_bound_ktree(None, current_depth + 1))

        return node

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
        for node in self.nodes:
            if not node.bound:
                self.solver.add(node.var != prod.id)

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
            if child.lhs == parent.rhs[x] or parent.rhs[x].name == 'Any':
                child_pos.append(x)

        for n in self.nodes:
            # not a leaf node
            if n.children:
                ctr_children = []
                for p in child_pos:
                    ctr_children.append(n.children[p].var == child.id)

                self.solver.add(self.smt_namespace.Implies(self.smt_namespace.Or(ctr_children), n.var != parent.id))

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
                else:
                    logger.warning('Predicate not handled: {}'.format(pred))
        except (KeyError, ValueError) as e:
            msg = 'Failed to resolve predicates. {}'.format(e)
            raise RuntimeError(msg) from None

    def blocking_template_compute(self, root=None) -> ExprRef:
        ctr = []
        for n in self.nodes:
            ctr.append(n.var != self.smt_namespace.Var(self.template_var_counter, self.smt_namespace.IntSort()))
            self.template_var_counter += 1
        return self.smt_namespace.Or(*ctr)
        # if root is None:
        #     return Or([self.blocking_template_compute(root) for root in self.roots])
        # ctr = root.var != Var(self.template_var_counter, IntSort())
        # self.template_var_counter += 1
        # if root.children:
        #     return Or(ctr, self.blocking_template_compute(root.children[0]),
        #                  self.blocking_template_compute(root.children[1]))
        # return ctr

    def model_values(self, model, root=None):
        """"Returns the input model, plus all models obtained through commutation of operations"""
        return [model[n.var] for n in self.nodes]
        # if root is None:
        #     for val_tuples in itertools.product(*[list(self.model_values(model, root)) for root in self.roots]):
        #         for val_tuples_perm in itertools.permutations(val_tuples):
        #             yield list(itertools.chain.from_iterable(val_tuples_perm))
        #     return
        #
        # acc = [model[root.var]]
        # if root.children:
        #     root_val = model[root.var].as_long()
        #     for m_values_0 in self.model_values(model, root.children[0]):
        #         for m_values_1 in self.model_values(model, root.children[1]):
        #             if root_val in self.commutative_prod_ids:  # yield original model + permutation models
        #                 yield acc + m_values_0 + m_values_1
        #                 yield acc + m_values_1 + m_values_0
        #             else:  # yield just original model
        #                 yield acc + m_values_0 + m_values_1
        # else:
        #     yield acc

    def block_model(self):
        self.solver.add(self.smt_namespace.substitute_vars(self.blocking_template, *self.model_values(self.model)))
        # for m_vals in self.model_values(self.model):
        #     perf.counter_inc(perf.BLOCK_PROGRAMS)

    def update(self, info=None):
        # if info is not None and not isinstance(info, str):
        #     for core in info:
        #         ctr = []
        #         for constraint in core:
        #             ctr.append(self.variables[self.program2tree[constraint[0]].id - 1] != constraint[1].id)
        #         self.solver.add(Or(*ctr))
        # else:
        self.block_model()

    def build_program(self, root=None):
        if root is None:
            return ApplyNode(self.and_production, [self.build_program(root) for root in self.roots])

        prod = self.spec.get_production_or_raise(self.model[root.var].as_long())

        if prod.is_function():
            return ApplyNode(prod, [self.build_program(root.children[i]) for i in range(len(prod.rhs))])
        elif prod.is_enum():
            return AtomNode(prod)
        else:
            raise NotImplementedError()

    def next(self):
        perf.timer_start(perf.Z3_ENUM_TIME)
        res = self.solver.check()
        perf.timer_stop(perf.Z3_ENUM_TIME)

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
            perf.timer_start(perf.BUILD_PROG_TIME)
            program = self.build_program()
            perf.timer_stop(perf.BUILD_PROG_TIME)
            return program
        else:
            return None
