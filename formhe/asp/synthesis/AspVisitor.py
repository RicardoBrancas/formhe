import clingo.ast
from ordered_set import OrderedSet

from formhe.trinity.DSL import TyrellSpec, EnumType, Node
from formhe.trinity.Visitor import Builder


class AspVisitor:

    def __init__(self, spec: TyrellSpec):
        self.spec = spec
        self.builder = Builder(spec)
        self.free_vars = OrderedSet()

    def visit(self, ast: clingo.ast.AST) -> Node:
        '''
        Dispatch to a visit method in a base class or visit and transform the
        children of the given AST if it is missing.
        '''
        attr = 'visit_' + str(ast.ast_type).replace('ASTType.', '')
        if hasattr(self, attr):
            return getattr(self, attr)(ast)
        else:
            raise NotImplementedError(attr)

    def visit_BooleanConstant(self, boolean_constant):
        if boolean_constant.value == 0:
            return self.builder.make_enum('Bool', False)
        else:
            return self.builder.make_enum('Bool', True)

    def visit_SymbolicAtom(self, symbolic_atom):
        return self.visit(symbolic_atom.symbol)

    def visit_Literal(self, literal):
        if literal.sign == 0:
            return self.visit(literal.atom)
        else:
            return self.builder.make_apply('not', [self.visit(literal.atom)])

    def visit_Variable(self, variable):
        Int: EnumType = self.spec.get_type('Int')
        if variable.name not in Int.domain:
            Int.domain.append(variable.name)
            self.spec._prod_spec.add_enum_production(Int, len(Int.domain) - 1, False)
            if variable.name == '_':
                self.spec.add_predicate('not_occurs', [(Int, '_')])
            self.free_vars.append(variable.name)
        return self.builder.make_enum('Int', variable.name)

    def visit_Function(self, function):
        args = [self.visit(a) for a in function.arguments]
        if function.name:
            return self.builder.make_apply(function.name, args)
        else:
            raise NotImplementedError()
            # return self.builder.make_apply('tuple', args)

    def visit_Comparison(self, comparison):
        match comparison.comparison:
            case clingo.ast.ComparisonOperator.Equal:
                return self.builder.make_apply('eq', [self.visit(comparison.left), self.visit(comparison.right)])
            case clingo.ast.ComparisonOperator.NotEqual:
                if comparison.left.ast_type == clingo.ast.ASTType.Function and comparison.left.name == '' and \
                        comparison.right.ast_type == clingo.ast.ASTType.Function and comparison.right.name == '':  # tuple comparison
                    comparison_elements = []
                    for elem_a, elem_b in zip(comparison.left.arguments, comparison.right.arguments):
                        comparison_elements.append(self.builder.make_apply('neq', [self.visit(elem_a), self.visit(elem_b)]))
                    return self.builder.make_apply('and', comparison_elements)
                else:
                    return self.builder.make_apply('neq', [self.visit(comparison.left), self.visit(comparison.right)])
            case _:
                raise NotImplementedError()

    def visit_BinaryOperation(self, binary_operation):
        match binary_operation.operator_type:
            case clingo.ast.BinaryOperator.Plus:
                return self.builder.make_apply('add', [self.visit(binary_operation.left), self.visit(binary_operation.right)])
            case _:
                raise NotImplementedError()

    def visit_Rule(self, rule):
        # head = self.visit(rule.head)
        body = [self.visit(a) for a in rule.body]
        # tree = self.builder.make_apply('and', [self.builder.make_apply('not', [head])] + body)
        return body
