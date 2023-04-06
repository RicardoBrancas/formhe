import clingo.ast

from formhe.asp.instance import Instance
from formhe.trinity.DSL import TyrellSpec, TypeSpec, ValueType, EnumType, ProductionSpec, PredicateSpec, ProgramSpec


class ASPSpecGenerator:

    def __init__(self, instance: Instance, free_vars: int, predicates: list = None) -> None:
        self.instance = instance
        self.free_vars_n = free_vars
        self.free_vars = []

        if predicates is None:
            self.predicates = self.instance.constantCollector.predicates.items()
        else:
            self.predicates = predicates

        self.type_spec = self.generate_type_spec()
        self.prod_spec = self.generate_prod_spec()
        self.pred_spec = self.generate_pred_spec()
        # self.prog_spec = ProgramSpec('asp', [], (self.Bool, self.PBool, self.Stmt))
        self.prog_spec = ProgramSpec('asp', [], (self.Bool, self.PBool))
        self.trinity_spec = TyrellSpec(self.type_spec, self.prog_spec, self.prod_spec, self.pred_spec)

        for i in range(self.free_vars_n):
            self.Terminal.domain.append('I' + str(i))
            self.prod_spec.add_enum_production(self.Terminal, len(self.Terminal.domain) - 1, False)
            self.free_vars.append('I' + str(i))
            # self.Bool.domain.append('B' + str(i))
            # prod_spec.add_enum_production(self.Bool, len(self.Bool.domain) - 1, False)

    def generate_type_spec(self) -> TypeSpec:
        type_spec = TypeSpec()
        self.Empty = ValueType('Empty')
        type_spec.define_type(self.Empty)
        self.Stmt = EnumType('Stmt')
        type_spec.define_type(self.Stmt)
        self.Aggregate = EnumType('Aggregate')
        type_spec.define_type(self.Aggregate)
        self.Bool = EnumType('Bool')
        type_spec.define_type(self.Bool)
        self.PBool = EnumType('PBool', [True, False])
        type_spec.define_type(self.PBool)
        self.Terminal = EnumType('Terminal', ['_', 0, 1] + self.instance.definitions)
        type_spec.define_type(self.Terminal)
        self.Interval = EnumType('Interval')
        type_spec.define_type(self.Interval)
        self.Pool = EnumType('Pool')
        type_spec.define_type(self.Pool)
        self.Any = EnumType('Any')
        type_spec.define_type(self.Any)
        self.BodyAggregateFunc = EnumType('BodyAggregateFunc', [func.name for func in clingo.ast.AggregateFunction])  # 'Count', 'Max', 'Min', 'Sum', 'SumPlus'
        type_spec.define_type(self.BodyAggregateFunc)
        return type_spec

    def generate_prod_spec(self) -> ProductionSpec:
        prod_spec = ProductionSpec()

        prod_spec.add_func_production('empty', self.Empty, [self.Empty], constant_expr=True)

        prod_spec.add_func_production('stmt', self.Stmt, [self.Any, self.Any], constant_expr=True)
        prod_spec.add_func_production('minimize', self.Stmt, [self.Terminal, self.Terminal, self.Any, self.Any], constant_expr=True)
        prod_spec.add_func_production('aggregate', self.Aggregate, [(self.Terminal, self.Empty), self.PBool, self.PBool, (self.Terminal, self.Empty)], constant_expr=True)
        prod_spec.add_func_production('aggregate_term', self.Aggregate, [(self.Terminal, self.Empty), self.Terminal, self.PBool, (self.Terminal, self.Empty)], constant_expr=True)
        prod_spec.add_func_production('aggregate_pool', self.Aggregate, [(self.Terminal, self.Empty), self.Pool, (self.Terminal, self.Empty)], constant_expr=True)
        prod_spec.add_func_production('body_aggregate', self.PBool, [self.BodyAggregateFunc, self.Aggregate], constant_expr=True)

        prod_spec.add_func_production('and', self.Bool, [self.Any, self.Any], constant_expr=True)
        prod_spec.add_func_production('and_', self.PBool, [self.Any, self.Any], constant_expr=True)
        prod_spec.add_func_production('or', self.PBool, [self.Any, self.Any], constant_expr=True)
        prod_spec.add_func_production('stmt_and', self.Bool, [self.Any, self.Any], constant_expr=True)
        prod_spec.add_func_production('tuple', self.Terminal, [self.Any], constant_expr=True)

        prod_spec.add_func_production('add', self.Terminal, [self.Terminal, self.Terminal], constant_expr=True)
        prod_spec.add_func_production('sub', self.Terminal, [self.Terminal, self.Terminal], constant_expr=True)
        prod_spec.add_func_production('mul', self.Terminal, [self.Terminal, self.Terminal], constant_expr=True)
        prod_spec.add_func_production('div', self.Terminal, [self.Terminal, self.Terminal], constant_expr=True)
        prod_spec.add_func_production('abs', self.Terminal, [self.Terminal], constant_expr=True)

        prod_spec.add_func_production('eq', self.Bool, [self.Terminal, self.Terminal], constant_expr=True)
        prod_spec.add_func_production('neq', self.Bool, [self.Terminal, self.Terminal], constant_expr=True)
        # prod_spec.add_func_production('and', self.Bool, [self.Bool, self.Bool], constant_expr=True)

        prod_spec.add_func_production('not', self.PBool, [self.PBool], constant_expr=True)
        prod_spec.add_func_production('classical_not', self.PBool, [self.PBool], constant_expr=True)

        prod_spec.add_func_production('pool', self.Pool, [self.Any, self.Any], constant_expr=False)
        prod_spec.add_func_production('interval', self.Interval, [self.Terminal, self.Terminal], constant_expr=False)

        for predicate_name, predicate_num in self.predicates:
            prod_spec.add_func_production(predicate_name, self.PBool, [self.Any] * predicate_num)

        return prod_spec

    def generate_pred_spec(self):
        pred_spec = PredicateSpec()

        pred_spec.add_predicate('not_occurs', ['and'])
        pred_spec.add_predicate('not_occurs', ['minimize'])
        pred_spec.add_predicate('not_occurs', ['and_'])
        pred_spec.add_predicate('not_occurs', ['or'])
        pred_spec.add_predicate('not_occurs', ['stmt_and'])
        pred_spec.add_predicate('not_occurs', ['stmt'])
        # pred_spec.add_predicate('not_occurs', ['aggregate'])
        pred_spec.add_predicate('not_occurs', ['tuple'])
        pred_spec.add_predicate('not_occurs', ['pool'])
        if not self.instance.config.enable_arithmetic:
            pred_spec.add_predicate('not_occurs', ['sub'])
            pred_spec.add_predicate('not_occurs', ['abs'])
            pred_spec.add_predicate('not_occurs', ['add'])
            pred_spec.add_predicate('not_occurs', ['mul'])
            pred_spec.add_predicate('not_occurs', ['div'])
        if self.instance.config.disable_classical_negation:
            pred_spec.add_predicate('not_occurs', ['classical_not'])
        pred_spec.add_predicate('not_occurs', ['aggregate_term'])
        pred_spec.add_predicate('not_occurs', ['aggregate_pool'])
        pred_spec.add_predicate('not_occurs', [(self.PBool, True)])
        pred_spec.add_predicate('not_occurs', [(self.PBool, False)])
        for fun in clingo.ast.AggregateFunction:
            pred_spec.add_predicate('not_occurs', [(self.BodyAggregateFunc, fun.name)])

        # pred_spec.add_predicate('commutative', ['and'])
        # pred_spec.add_predicate('commutative', ['or'])
        pred_spec.add_predicate('commutative', ['add'])
        pred_spec.add_predicate('commutative', ['mul'])
        pred_spec.add_predicate('commutative', ['eq'])
        pred_spec.add_predicate('commutative', ['neq'])

        for predicate_name, _ in self.predicates:
            for p in self.prod_spec.get_function_productions():
                if p.name != 'interval':
                    pred_spec.add_predicate('is_not_parent', [predicate_name, p.name])

        for p in self.prod_spec.get_function_productions():
            if p.name not in list(map(lambda x: x[0], self.predicates)):
                pred_spec.add_predicate('is_not_parent', [p.name, (self.Terminal, '_')])

        # for p in self.prod_spec.get_function_productions():
        #     pred_spec.add_predicate('is_not_parent', [p.name, 'stmt'])

        for i in self.Terminal.domain:
            if isinstance(i, int) and i < 0:
                for j in self.Terminal.domain:
                    if isinstance(j, int) and -i == j:
                        pred_spec.add_predicate('is_not_parent', ['add', (self.Terminal, i)])
                        pred_spec.add_predicate('is_not_parent', ['sub', (self.Terminal, i)])

        pred_spec.add_predicate('is_not_parent', ['add', (self.Terminal, 0)])
        pred_spec.add_predicate('is_not_parent', ['sub', (self.Terminal, 0)])
        pred_spec.add_predicate('is_not_parent', ['div', (self.Terminal, 0)])
        pred_spec.add_predicate('is_not_parent', ['div', (self.Terminal, 1), [1]])
        # pred_spec.add_predicate('is_not_parent', ['div', (self.Terminal, -1), [1]])
        pred_spec.add_predicate('is_not_parent', ['mul', (self.Terminal, 0)])
        pred_spec.add_predicate('is_not_parent', ['mul', (self.Terminal, 1)])

        pred_spec.add_predicate('distinct_args', ['eq'])
        pred_spec.add_predicate('distinct_args', ['neq'])
        pred_spec.add_predicate('distinct_args', ['interval'])
        pred_spec.add_predicate('distinct_args', ['sub'])
        pred_spec.add_predicate('distinct_args', ['div'])

        return pred_spec
