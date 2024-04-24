import formhe

from formhe.trinity.z3_enumerator import Z3Enumerator
from formhe.trinity.DSL import TyrellSpec, ApplyNode, AtomNode
from formhe.trinity.smt_enumerator import Statement, Solver, Node
from trinity.smt_enumerator import PresetStatement


class MutationNode(Node):

    def bind(self, production_id: int, enumerator: 'SmtEnumerator', parent: Node = None):
        self.bound = True
        self.mutation_counter = enumerator.create_variable(f'node_{self.id}_bind_counter', enumerator.smt_namespace.Bool)
        # enumerator.create_assertion((self.var == production_id) != self.mutation_counter, f'node_{self.id}_binding', True)
        empty_prod = enumerator.spec.get_function_production('empty').id
        tmp = enumerator.smt_namespace.Or(self.var == production_id,
                                     enumerator.smt_namespace.And(parent.var == empty_prod, self.var == empty_prod) if parent is not None else False,
                                     self.mutation_counter)
        enumerator.create_assertion(tmp, f'node_{self.id}_binding', True, True)


class MutationStatement(Statement):

    def __init__(self, enumerator: 'SmtEnumerator', statement: PresetStatement):
        self.id = enumerator.statement_counter
        enumerator.statement_counter += 1
        self.nodes = []
        self.head_nodes = []
        self.body_nodes = []

        if statement.has_head:
            if statement.head is None:
                raise ValueError()
            else:
                self.head = self.create_semi_bound_tree(enumerator, statement.head, is_head=True)
        else:
            self.head = self.create_semi_bound_tree(enumerator, None, is_head=True)

        self.body = []
        for preset_atom in statement.body:
            if preset_atom is not None:
                self.body.append(self.create_semi_bound_tree(enumerator, preset_atom, is_body=True))
            else:
                raise ValueError()

        self.create_head_output_constraints(enumerator, head_nullable=not statement.has_head)
        self.create_body_output_constraints(enumerator)
        for root in self.body:
            self.create_aggregate_restrictions_constraints(enumerator, root)
        for child in self.head.children:
            self.create_aggregate_restrictions_constraints(enumerator, child)
        self.create_children_constraints(enumerator)
        # if config.get().block_constant_expressions:
        #     self.create_no_constant_sub_expressions_constraints(enumerator)
        self.create_head_empty_or_non_constant_constraints(enumerator)
        self.create_no_dont_care_in_head_constraints(enumerator)
        self.create_no_constant_aggregate_constraints(enumerator)

    def create_semi_bound_tree(self, enumerator: 'SmtEnumerator', ast_node: formhe.trinity.DSL.Node, is_body: bool = False, is_head: bool = False, parent: Node = None) -> MutationNode:
        node = MutationNode(enumerator, enumerator.node_counter, 0, False, enumerator.max_children_except_aggregate)
        self.nodes.append(node)
        if is_body:
            self.body_nodes.append(node)
        if is_head:
            self.head_nodes.append(node)
        enumerator.node_counter += 1
        if ast_node is None:
            node.bind(0, enumerator, parent)
        elif isinstance(ast_node, ApplyNode):
            node.bind(ast_node.production.id, enumerator, parent)
            for child in ast_node.children:
                node.children.append(self.create_semi_bound_tree(enumerator, child, is_body, is_head, node))
        elif isinstance(ast_node, AtomNode):
            node.bind(ast_node.production.id, enumerator, parent)

        return node


class BugEnumerator(Z3Enumerator):

    def __init__(self, spec: TyrellSpec, depth=None, predicates_names=None, free_vars=None, statements=None):
        if predicates_names is None:
            predicates_names = []
        if statements is None:
            statements = []

        self.solver: Solver = self.init_solver()

        self.predicate_names = predicates_names
        # self.cores = cores
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

        self.statements = [MutationStatement(self, statement) for statement in statements]

        # print(self.statements)

        # for node in self.nodes:
        #     print('   ' * (node.depth - 1) + str(node))
        self.model = None
        # self.create_input_constraints()

        self.commutative_prod_ids = []

        self.resolve_predicates()

        for statement in self.statements:
            statement.finish_init(self)

        # if strict_minimum_depth:
        #     self.create_max_depth_used_constraints()
        # self.create_semantic_constraints()

        # if not config.get().allow_not_generated_predicates:
        #     self.create_predicate_usage_constraints(free_predicates)
        #     self.create_predicate_force_generate_constraints(force_generate_predicates)

        self.blocking_template = self.blocking_template_compute()

        # print(self.spec._prod_spec)

        self.and_production = self.spec.get_function_production('and')
        self.stmt_and_production = self.spec.get_function_production('stmt_and')
        self.stmt_production = self.spec.get_function_production('stmt')
        self.has_enumerated = False

    def set_mutation_count(self, i: int):
        mutation_vars = []
        for stmt in self.statements:
            for node in stmt.nodes:
                if isinstance(node, MutationNode):
                    mutation_vars.append(node.mutation_counter)

        self.create_assertion(self.smt_namespace.Sum(*mutation_vars) == i, "mutation_count_bound", debug_print=True)
