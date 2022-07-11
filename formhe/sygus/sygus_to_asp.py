import itertools
import string
from collections import defaultdict

import cvc5

precedence = defaultdict(lambda: 100)
precedence.update({
    'VARIABLE': -100,
    'CONST_RATIONAL': -100,
    'ABS': -100,
    'APPLY_CONSTRUCTOR': -100,
    'MUL': 0,
    'DIV': 0,
    'MOD': 0,
    'ADD': 1,
    'SUB': 1,
    'LT': 2,
    'LE': 2,
    'GT': 2,
    'GE': 2,
    'EQUAL': 3,
    'NOT': 4,
    'OR': 5
})


def l_paren(kind_parent, kind_child):
    if precedence[kind_parent] <= precedence[kind_child]:
        return '('
    return ''


def r_paren(kind_parent, kind_child):
    if precedence[kind_parent] <= precedence[kind_child]:
        return ')'
    return ''


def BinaryTerm(kind, symbol):
    def inner(clazz):
        type.__setattr__(clazz, f'visit_term_{kind}', lambda self,
                                                             term: f'{l_paren(kind, term[0].getKind().name)}{self(term[0])}{r_paren(kind, term[0].getKind().name)} {symbol} {l_paren(kind, term[1].getKind().name)}{self(term[1])}{r_paren(kind, term[1].getKind().name)}')
        return clazz

    return inner


@BinaryTerm('EQUAL', '==')
@BinaryTerm('ADD', '+')
@BinaryTerm('SUB', '-')
@BinaryTerm('MUL', '*')
@BinaryTerm('DIV', '/')
@BinaryTerm('GT', '>')
@BinaryTerm('LT', '<')
@BinaryTerm('LE', '<=')
@BinaryTerm('GE', '>=')
@BinaryTerm('OR', '|')
class SyGuSToASP:

    def __init__(self, sym_by_vars, vars_by_sym):
        self.sym_by_vars = sym_by_vars
        self.vars_by_sym = vars_by_sym
        self.letter_counter = 0
        self.var_letter_mapper = {}
        self.var_letters = defaultdict(list)
        self.var_groups = defaultdict(list)
        self.inits = []
        self.initted = set()

    def __call__(self, obj, *args, **kwargs):
        return self.visit(obj, *args, **kwargs)

    def reset_state(self):
        self.letter_counter = 0
        self.var_letter_mapper = {}
        self.var_letters = defaultdict(list)
        self.var_groups = defaultdict(list)
        self.inits = []
        self.initted = set()

    def visit_statement(self, obj):
        self.reset_state()
        conditions = self(obj)
        guards = []
        for sym, groups in self.var_groups.items():
            for combo1, combo2 in itertools.combinations(groups, 2):
                guard1 = '(' + ', '.join(combo1) + ')'
                guard2 = '(' + ', '.join(combo2) + ')'
                guards.append(f'{guard1} != {guard2}')

        if guards:
            guards_str = ', '.join(guards) + ', '
        else:
            guards_str = ''

        return f':- ' + ', '.join(self.inits) + ', ' + guards_str + conditions + '.'

    def visit(self, obj, *args, **kwargs):
        if isinstance(obj, cvc5.Term):
            method_name = 'visit_term_' + obj.getKind().name
            method = self.__getattribute__(method_name)
            return method(obj, *args, **kwargs)
        else:
            raise NotImplementedError(f'Visit method for {str(obj)} not implemented!')

    def visit_term_ABS(self, abs_term):
        return f'abs({self(abs_term[0])})'

    def visit_term_NOT(self, abs_term):
        if abs_term[0].getKind() == cvc5.Kind.EQUAL:
            return f'{self(abs_term[0][0])} != {self(abs_term[0][1])}'
        else:

            return f'-{self(abs_term[0])}'

    def visit_term_AND(self, and_term):
        return f'{self(and_term[0])}, {self(and_term[1])}'

    def visit_term_VARIABLE(self, var_term):
        sym, sym_i, arg_i = self.sym_by_vars[var_term]
        if (sym, sym_i) not in self.initted:
            for arg_j in range(len(self.vars_by_sym[sym][sym_i])):
                if (sym, arg_j) not in self.var_letter_mapper:
                    self.var_letter_mapper[(sym, arg_j)] = string.ascii_uppercase[self.letter_counter]
                    self.var_letters[sym].append(string.ascii_uppercase[self.letter_counter])
                    self.letter_counter += 1
            self.var_groups[sym].append([letter + str(sym_i+1) for letter in self.var_letters[sym]])
            self.initted.add((sym, sym_i))
            self.inits.append(f'{sym}({", ".join([letter + str(sym_i+1) for letter in self.var_letters[sym]])})')

        letter = self.var_letter_mapper[(sym, arg_i)]

        return f'{letter}{sym_i + 1}'

    def visit_term_CONST_RATIONAL(self, const_term):
        return str(const_term.getRealValue())

    def visit_term_APPLY_CONSTRUCTOR(self, apply_constructor_term):
        return str(apply_constructor_term)
