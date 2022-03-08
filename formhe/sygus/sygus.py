from collections import defaultdict
from typing import Dict, Set, List, Union, Tuple, Collection

import pycvc5
from ordered_set import OrderedSet

Grammar = Dict[pycvc5.Term, OrderedSet[pycvc5.Term]]


class SyGuSProblem:
    grammar: Grammar
    constraints: List[pycvc5.Term]
    starting_symbol = Union[None, pycvc5.Term]
    inputs: List[pycvc5.Term]

    def __init__(self, fun_name: str, debug: bool = False):
        self.fun_name = fun_name
        self.grammar = defaultdict(OrderedSet)
        self.inputs = list()
        self.inputs_by_sort = defaultdict(OrderedSet)
        self.constraints = list()
        self.starting_symbol = None
        self.function = None
        self.debug = debug
        self.constantSort = None

    def add_grammar_rule(self, var: pycvc5.Term, term: pycvc5.Term):
        self.grammar[var].add(term)

    def set_grammar_rules(self, rules: Dict[pycvc5.Term, Set[Union[Tuple, pycvc5.Term]]], solver: Union[None, pycvc5.Solver] = None):
        for var, terms in rules.items():
            for term in terms:
                if isinstance(term, tuple) or isinstance(term, list):
                    term = solver.mkTerm(*term)
                elif isinstance(term, int):
                    term = solver.mkInteger(term)

                self.add_grammar_rule(var, term)

    def add_constraint(self, constraint: pycvc5.Term):
        self.constraints.append(constraint)

    def realize_constraints(self, solver: pycvc5.Solver):
        for constraint in self.constraints:
            solver.addSygusConstraint(constraint)

    def set_starting_symbol(self, var: pycvc5.Term):
        self.starting_symbol = var

    def add_input(self, sort: pycvc5.Sort, var: pycvc5.Term):
        self.inputs.append(var)
        self.inputs_by_sort[sort].append(var)

    def get_synth_fun(self, solver: pycvc5.Solver):
        self.function = solver.synthFun(self.fun_name, self.inputs, self.starting_symbol.getSort(), self.get_grammar(solver))
        return self.function

    def get_grammar(self, solver):
        g = solver.mkSygusGrammar(self.inputs, self.grammar.keys())

        for lhs, rhs in self.grammar.items():
            print(lhs, rhs)
            g.addRules(lhs, rhs)

        return g

    def make_enum(self, solver: pycvc5.Solver, values: Collection[str], name='constant'):
        self.constructors = {}
        for value in values:
            constructor = solver.mkDatatypeConstructorDecl(value)
            self.constructors[value] = constructor
        self.constantSort = solver.declareDatatype(name, list(self.constructors.values()))
        return self.constantSort

    def enumerate(self, solver: pycvc5.Solver, max_sols=None):
        sols_enumerated = 0
        if max_sols == sols_enumerated or not solver.checkSynth().isUnsat():
            raise StopIteration()
        else:
            sols_enumerated += 1
            yield solver.getSynthSolutions([self.function])[0][1]
        while (max_sols is None or sols_enumerated < max_sols) and solver.checkSynthNext().isUnsat():
            sols_enumerated += 1
            yield solver.getSynthSolutions([self.function])[0][1]

    def __str__(self):
        res = f'''
(synth-fun {self.fun_name} ({' '.join(map(lambda i: f'({i.getSymbol()} {i.getSort()})', self.inputs))}) {self.starting_symbol.getSort()}
    ({' '.join(map(lambda s: f'({s.getSymbol()} {s.getSort()})', self.grammar.keys()))})
    ('''
        for var, terms in self.grammar.items():
            res += f'({var.getSymbol()} {var.getSort()} ({" ".join(map(str, terms))}))\n    '
        res = res[:-4] + ')\n'
        for constraint in self.constraints:
            res += f'(constraint {str(constraint)})\n'
        return res
