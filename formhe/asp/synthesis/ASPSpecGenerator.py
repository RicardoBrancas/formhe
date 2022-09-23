from formhe.asp.instance import Instance
from formhe.trinity.DSL import TyrellSpec, TypeSpec, ValueType, EnumType, ProductionSpec, PredicateSpec, ProgramSpec


class ASPSpecGenerator:

    def __init__(self, instance: Instance, free_vars: int) -> None:
        self.instance = instance
        self.free_vars = free_vars

        self.type_spec = self.generate_type_spec()
        self.prod_spec = self.generate_prod_spec()
        self.pred_spec = self.generate_pred_spec()
        self.prog_spec = ProgramSpec('asp', [], (self.Bool, self.PBool))
        self.trinity_spec = TyrellSpec(self.type_spec, self.prog_spec, self.prod_spec, self.pred_spec)

    def generate_type_spec(self) -> TypeSpec:
        type_spec = TypeSpec()
        self.Empty = ValueType('Empty')
        type_spec.define_type(self.Empty)
        self.Bool = EnumType('Bool')
        type_spec.define_type(self.Bool)
        self.PBool = EnumType('PBool', [True, False])
        type_spec.define_type(self.PBool)
        self.Int = EnumType('Int', [-1, 0, 1])
        type_spec.define_type(self.Int)
        self.Any = EnumType('Any')
        type_spec.define_type(self.Any)
        return type_spec

    def generate_prod_spec(self) -> ProductionSpec:
        prod_spec = ProductionSpec()

        prod_spec.add_func_production('empty', self.Empty, [self.Empty], constant_expr=True)

        prod_spec.add_func_production('and', self.Bool, [self.Any, self.Any], constant_expr=True)
        prod_spec.add_func_production('tuple', self.Int, [self.Any], constant_expr=True)

        prod_spec.add_func_production('add', self.Int, [self.Int, self.Int], constant_expr=True)
        prod_spec.add_func_production('sub', self.Int, [self.Int, self.Int], constant_expr=True)
        prod_spec.add_func_production('mul', self.Int, [self.Int, self.Int], constant_expr=True)
        prod_spec.add_func_production('div', self.Int, [self.Int, self.Int], constant_expr=True)
        prod_spec.add_func_production('abs', self.Int, [self.Int], constant_expr=True)

        prod_spec.add_func_production('eq', self.Bool, [self.Int, self.Int], constant_expr=True)
        prod_spec.add_func_production('neq', self.Bool, [self.Int, self.Int], constant_expr=True)
        # prod_spec.add_func_production('and', self.Bool, [self.Bool, self.Bool], constant_expr=True)
        # prod_spec.add_func_production('or', self.Bool, [self.Bool, self.Bool], constant_expr=True)
        prod_spec.add_func_production('not', self.PBool, [self.PBool], constant_expr=True)
        prod_spec.add_func_production('classical_not', self.PBool, [self.PBool], constant_expr=True)

        for predicate_name, predicate_num in self.instance.constantCollector.predicates.items():
            prod_spec.add_func_production(predicate_name, self.PBool, [self.Any] * predicate_num)

        # free variable productions

        # for i in range(self.free_vars):
        #     self.Int.domain.append('I' + str(i))
        #     prod_spec.add_enum_production(self.Int, len(self.Int.domain) - 1, False)
        #     #self.Bool.domain.append('B' + str(i))
        #     #prod_spec.add_enum_production(self.Bool, len(self.Bool.domain) - 1, False)

        return prod_spec

    def generate_pred_spec(self):
        pred_spec = PredicateSpec()

        pred_spec.add_predicate('not_occurs', ['and'])
        pred_spec.add_predicate('not_occurs', ['tuple'])
        pred_spec.add_predicate('not_occurs', [(self.PBool, True)])
        pred_spec.add_predicate('not_occurs', [(self.PBool, False)])

        # pred_spec.add_predicate('commutative', ['and'])
        # pred_spec.add_predicate('commutative', ['or'])
        pred_spec.add_predicate('commutative', ['add'])
        pred_spec.add_predicate('commutative', ['mul'])
        pred_spec.add_predicate('commutative', ['eq'])
        pred_spec.add_predicate('commutative', ['neq'])

        for predicate_name in self.instance.constantCollector.predicates.keys():
            for p in self.prod_spec.get_function_productions():
                if p.name != 'or' and p.name != 'and' and p.name != 'empty':
                    pred_spec.add_predicate('is_not_parent', [predicate_name, p.name])

        for predicate_name in self.instance.constantCollector.predicates.keys():
            for p in self.prod_spec.get_function_productions():
                pred_spec.add_predicate('is_not_parent', [predicate_name, p.name])

        for i in self.Int.domain:
            if i < 0:
                for j in self.Int.domain:
                    if -i == j:
                        pred_spec.add_predicate('is_not_parent', ['add', (self.Int, i)])
                        pred_spec.add_predicate('is_not_parent', ['sub', (self.Int, i)])

        pred_spec.add_predicate('is_not_parent', ['add', (self.Int, 0)])
        pred_spec.add_predicate('is_not_parent', ['sub', (self.Int, 0)])
        pred_spec.add_predicate('is_not_parent', ['div', (self.Int, 0)])
        pred_spec.add_predicate('is_not_parent', ['div', (self.Int, 1), [1]])
        pred_spec.add_predicate('is_not_parent', ['div', (self.Int, -1), [1]])
        pred_spec.add_predicate('is_not_parent', ['mul', (self.Int, 0)])
        pred_spec.add_predicate('is_not_parent', ['mul', (self.Int, 1)])

        return pred_spec
