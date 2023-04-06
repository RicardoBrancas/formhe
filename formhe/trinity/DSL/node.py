import copy
from abc import abstractmethod, ABC
from itertools import chain
from typing import cast, List, Any, Optional

from formhe.trinity.DSL.spec import Type, Production, EnumProduction


class Node(ABC):
    """Generic and abstract AST Node"""
    # Each instance of class Node has its own id.
    current_id: int = 0

    @abstractmethod
    def __init__(self, prod: Production):
        self._prod = prod
        self.id = Node.current_id
        Node.current_id += 1
        self.z3_x_vars = []
        self.z3_cond_vars = []

    @property
    def production(self) -> Production:
        return self._prod

    @property
    def type(self) -> Type:
        return self._prod.lhs

    @abstractmethod
    def is_leaf(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_enum(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_param(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_apply(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def children(self) -> List['Node']:
        raise NotImplementedError

    def depth(self):
        """ Returns the maximum depth of the program. """
        if not self.has_children():
            return 1
        else:
            # Compute the depth of each subtree
            children_depths = map(lambda ch: ch.depth(), self.children)

            # Use the larger one
            return 1 + max(children_depths)

    def has_children(self):
        """ Returns True if the node has children. """
        return self.children is not None and len(self.children) > 0

    def get_subtree(self):
        """ Return as an ordered list all the descendant nodes """
        if not self.has_children():
            return [self]
        else:
            return [self] + list(chain(*map(lambda c: c.get_subtree(), self.children)))

    def get_subtree_name_w_depths(self):
        """ Return as an ordered list all the descendant nodes """
        if self.is_apply():
            name = self.production.name
        elif self.is_param():
            name = str(self.index)
        elif self.is_enum():
            name = "numvar"
        else:
            raise ValueError()
        if not self.has_children():
            return [(name, self.depth())]

        else:
            return [(name, self.depth())] + list(chain(*map(lambda c: c.get_subtree_name_w_depths(), self.children)))

    def count_vars(self):
        if self.type.name == "NumVar":
            return 1 + sum(map(lambda n: n.count_vars(), self.children))
        return sum(map(lambda n: n.count_vars(), self.children))

    def has_missing_vars(self):
        """ Returns True if there is a 'x' variable in any of the descendant nodes. """
        if self.type.name == "NumVar":
            return True
        elif self.has_children():
            return any(map(lambda c: c.has_missing_vars(), self.children))
        else:
            return False

    def is_complete(self):
        """ Returns True if there is no 'x' variable in any of the descendant nodes. """
        return not self.has_missing_vars()

    def needs_conditional(self):
        nodes = [self] + self.get_subtree()
        for node in nodes:
            if node.is_apply() and node.name == "if":
                return True
        return False

    def get_subtree_from_operation(self, operation) -> Optional["Node"]:
        """ Returns the node assigned 'operation' if it exists. """
        if self.is_apply() and self.name == operation:
            return self
        elif self.has_children():
            for c in self.children:
                subtree = c.get_subtree_from_operation(operation)
                if subtree is not None:
                    return subtree
            return None
        else:
            return None

    def __getstate__(self):
        new = copy.deepcopy(self.__dict__)
        if 'z3_x_vars' in new.keys():
            del new['z3_x_vars']
        if 'z3_cond_vars' in new.keys():
            del new['z3_cond_vars']
        return new

    def __setstate__(self, dct):
        dct['z3_x_vars'] = []
        dct['z3_cond_vars'] = []
        self.__dict__ = dct


class LeafNode(Node):
    """Generic and abstract class for AST nodes that have no children"""

    @abstractmethod
    def __init__(self, prod: Production):
        super().__init__(prod)
        if prod.is_function():
            raise ValueError(
                'Cannot construct an AST leaf node from a FunctionProduction')

    def is_leaf(self) -> bool:
        return True

    def is_apply(self) -> bool:
        return False


class AtomNode(LeafNode):
    """Leaf AST node that holds string data"""

    def __init__(self, prod: Production):
        if not prod.is_enum():
            raise ValueError(
                'Cannot construct an AST atom node from a non-enum production')
        super().__init__(prod)
        self.data = self.get_data()

    def get_data(self) -> Any:
        prod = cast(EnumProduction, self._prod)
        return prod.rhs[0]

    @property
    def children(self) -> List[Node]:
        return []

    def is_enum(self) -> bool:
        return True

    def is_param(self) -> bool:
        return False

    def deep_eq(self, other) -> bool:
        '''
        Test whether this node is the same with ``other``. This function performs deep comparison rather than just comparing the object identity.
        '''
        if isinstance(other, AtomNode):
            return self.type == other.type and self.data == other.data
        return False

    def deep_hash(self) -> int:
        '''
        This function performs deep hash rather than just hashing the object identity.
        '''
        return hash((self.type, str(self.data)))

    def __repr__(self) -> str:
        return 'AtomNode({})'.format(self.data)

    def __str__(self) -> str:
        return '{}'.format(self.data)


class ParamNode(LeafNode):
    """Leaf AST node that holds a param"""

    def __init__(self, prod: Production):
        if not prod.is_param():
            raise ValueError(
                'Cannot construct an AST param node from a non-param production')
        super().__init__(prod)

    @property
    def index(self) -> int:
        return self._prod.rhs[0]

    @property
    def children(self) -> List[Node]:
        return []

    def is_enum(self) -> bool:
        return False

    def is_param(self) -> bool:
        return True

    def deep_eq(self, other) -> bool:
        """
        Test whether this node is the same with ``other``. This function performs deep comparison rather than just comparing the object identity.
        """
        if isinstance(other, ParamNode):
            return self.index == other.index
        return False

    def deep_hash(self) -> int:
        '''
        This function performs deep hash rather than just hashing the object identity.
        '''
        return hash(self.index)

    def __repr__(self) -> str:
        return f'ParamNode({self.index})'

    def __str__(self) -> str:
        return f'@p{self.index}'


class ApplyNode(Node):
    """Internal AST node that represent function application"""

    def __init__(self, prod: Production, args: List[Node]):
        super().__init__(prod)
        if not prod.is_function():
            raise ValueError(
                'Cannot construct an AST internal node from a non-function production')
        for index, (decl_ty, node) in enumerate(zip(prod.rhs, args)):
            actual_ty = node.type
            if not isinstance(decl_ty, tuple):
                decl_ty = (decl_ty,)
            found_match = False
            for decl_ty_elem in decl_ty:
                if decl_ty_elem.name == 'Any' or decl_ty_elem == actual_ty:
                    found_match = True
                    break
            if not found_match:
                msg = f'Argument {index} type mismatch on {prod}: expected one of {decl_ty} but found {actual_ty}'
                raise ValueError(msg)
        self._args = args

    @property
    def name(self) -> str:
        return self._prod.name

    @property
    def args(self) -> List[Node]:
        return self._args

    @property
    def children(self) -> List[Node]:
        return self._args

    def is_leaf(self) -> bool:
        return False

    def is_enum(self) -> bool:
        return False

    def is_param(self) -> bool:
        return False

    def is_apply(self) -> bool:
        return True

    def deep_eq(self, other) -> bool:
        """
        Test whether this node is the same with ``other``. This function performs deep comparison rather than just comparing the object identity.
        """
        if isinstance(other, ApplyNode):
            return self.name == other.name and \
                   len(self.args) == len(other.args) and \
                   all(x.deep_eq(y)
                       for x, y in zip(self.args, other.args))
        return False

    def deep_hash(self) -> int:
        '''
        This function performs deep hash rather than just hashing the object identity.
        '''
        return hash((self.name, tuple(map(lambda x: x.deep_hash(), self.args))))

    def __repr__(self) -> str:
        return f'ApplyNode({self.name}, {self._args})'

    def __str__(self) -> str:
        return f'{self.name}({", ".join(map(str, self._args))})'
