from typing import Union, List

from formhe.trinity.DSL.node import AtomNode, Node, ParamNode, ApplyNode
from formhe.trinity.DSL.production import Production
from formhe.trinity.DSL.spec import TyrellSpec
from formhe.trinity.Visitor.generic_visitor import GenericVisitor


class ProductionVisitor(GenericVisitor):
    _children: List[Node]

    def __init__(self, children: List[Node]):
        self._children = children

    def visit_enum_production(self, prod) -> Node:
        return AtomNode(prod)

    def visit_param_production(self, prod) -> Node:
        return ParamNode(prod)

    def visit_function_production(self, prod) -> Node:
        return ApplyNode(prod, self._children)


class Builder:
    """ A factory class to build AST node """

    def __init__(self, spec: TyrellSpec):
        self._spec = spec

    def _make_node(self, prod: Production, children=None) -> Node:
        if children is None:
            children = []
        return ProductionVisitor(children).visit(prod)

    def make_node(self, src: Union[int, Production], children=None) -> Node:
        """
        Create a node with the given production index and children.
        Raise `KeyError` or `ValueError` if an error occurs
        """
        if children is None:
            children = []
        if isinstance(src, int):
            return self._make_node(self._spec.get_production_or_raise(src), children)
        elif isinstance(src, Production):
            # Sanity check first
            prod = self._spec.get_production_or_raise(src.id)
            if src != prod:
                raise ValueError(
                    'DSL Builder found inconsistent production instance')
            return self._make_node(prod, children)
        else:
            raise ValueError(
                f'make_node() only accepts int or production, but found {src}')

    def make_enum(self, name: str, value: str) -> Node:
        '''
        Convenient method to create an enum node.
        Raise `KeyError` or `ValueError` if an error occurs
        '''
        ty = self._spec.get_type_or_raise(name)
        prod = self._spec.get_enum_production_or_raise(ty, value)
        return self.make_node(prod.id)

    def make_param(self, index: int) -> Node:
        '''
        Convenient method to create a param node.
        Raise `KeyError` or `ValueError` if an error occurs
        '''
        prod = self._spec.get_param_production_or_raise(index)
        return self.make_node(prod.id)

    def make_apply(self, name: str, args: List[Node]) -> Node:
        '''
        Convenient method to create an apply node.
        Raise `KeyError` or `ValueError` if an error occurs
        '''
        prod = self._spec.get_function_production_or_raise(name)
        return self.make_node(prod.id, args)
