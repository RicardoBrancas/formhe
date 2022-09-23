from abc import ABC, abstractmethod
from typing import List, Any, cast

from .expr import Expr, ExprType
from .type import Type, EnumType, ValueType, LeafType


class Production(ABC):
    """
    This class represent a CFG production rule for our DSL.
    Each production rule is uniquely identified by its ID in a given spec.
    """

    _id: int
    _lhs: Type
    _constant_expr: bool

    @abstractmethod
    def __init__(self, id: int, lhs: Type, constant_expr: bool = False):
        self._id = id
        self._lhs = lhs
        self._constant_expr = constant_expr

    @property
    def id(self) -> int:
        return self._id

    @property
    def lhs(self) -> Type:
        return self._lhs

    @property
    def is_constant(self) -> bool:
        return self._constant_expr

    @property
    @abstractmethod
    def rhs(self) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def is_enum(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_param(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_function(self) -> bool:
        raise NotImplementedError


class EnumProduction(Production):
    _choice: int

    def __init__(self, id: int, lhs: EnumType, choice: int, constant_expr: bool = True):
        super().__init__(id, lhs, constant_expr)
        if choice >= len(lhs.domain):
            msg = 'Cannot create a EnumProduction with choice {} for a domain with {} elements.'.format(
                choice, len(lhs.domain))
            raise ValueError(msg)
        self._choice = choice

    def _get_rhs(self) -> Any:
        lhs_ty = cast(EnumType, self._lhs)
        return lhs_ty.domain[self._choice]

    @property
    def rhs(self) -> List[Any]:
        return [self._get_rhs()]

    def is_function(self) -> bool:
        return False

    def is_enum(self) -> bool:
        return True

    def is_param(self) -> bool:
        return False

    def __repr__(self) -> str:
        return 'EnumProduction(id={}, lhs={!r}, choice={})'.format(
            self._id, self._lhs, self._choice)

    def __str__(self) -> str:
        return 'Production {}: {} -> {}'.format(
            self._id, self._lhs, self._get_rhs())


class ParamProduction(Production):

    def __init__(self, id: int, lhs: ValueType, param_id: int, constant_expr: bool = False):
        super().__init__(id, lhs, constant_expr)
        if not isinstance(lhs, ValueType):
            raise ValueError('LHS of ParamProduction must be a value type')
        self._param_id = param_id

    @property
    def rhs(self) -> List[int]:
        return [self._param_id]

    def is_function(self) -> bool:
        return False

    def is_param(self) -> bool:
        return True

    def is_enum(self) -> bool:
        return False

    def __repr__(self) -> str:
        return 'ParamProduction(id={}, lhs={!r}, param_id={})'.format(
            self._id, self._lhs, self._param_id)

    def __str__(self) -> str:
        return 'Production {}: {} -> <param {}>'.format(
            self._id, self._lhs, self._param_id)


class FunctionProduction(Production):
    _name: str
    _rhs: List[Type]
    _constraints: List[Expr]

    def __init__(self, id: int, name: str, lhs: LeafType, rhs: List[Type], constraints=None, constant_expr: bool = False):
        super().__init__(id, lhs, constant_expr)
        if constraints is None:
            constraints = []
        if not isinstance(lhs, ValueType) and not isinstance(lhs, EnumType):
            raise ValueError('LHS of FunctionProduction must be a leaf type')
        if len(rhs) == 0:
            raise ValueError(
                'Cannot construct a FunctionProduction with empty RHS')
        for constraint in constraints:
            if constraint.type is not ExprType.BOOL:
                raise ValueError(
                    'Constraint does not have bool type: "{}"'.format(constraint))
        self._name = name
        self._rhs = rhs
        self._constraints = constraints

    @property
    def rhs(self) -> List[Type]:
        return self._rhs

    @property
    def name(self) -> str:
        return self._name

    @property
    def constraints(self) -> List[Expr]:
        return self._constraints

    def is_function(self) -> bool:
        return True

    def is_param(self) -> bool:
        return False

    def is_enum(self) -> bool:
        return False

    def __repr__(self) -> str:
        return 'FunctionProduction(id={}, lhs={!r}, name={}, rhs={})'.format(
            self._id, self._lhs, self._name, self._rhs)

    def __str__(self) -> str:
        return 'Production {}: {} -> {}({})'.format(
            self._id, self._lhs, self._name,
            ', '.join([str(x) for x in self._rhs]))
