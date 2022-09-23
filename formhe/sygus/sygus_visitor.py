import itertools
import logging
import string
from collections import defaultdict
from typing import Dict, List, Iterable, Collection

import clingo
import cvc5
from clingo.symbol import Symbol, SymbolType
from cvc5 import Kind

from formhe.asp.instance import Instance
from formhe.sygus.sygus import SyGuSProblem

logger = logging.getLogger('formhe.asp.sygus')

# def define_fun_to_string(f, params, body):
#     sort = f.getSort()
#     if sort.isFunction():
#         sort = f.getSort().getFunctionCodomainSort()
#     result = ""
#     result += "(define-fun " + str(f) + " ("
#     for i in range(0, len(params)):
#         if i > 0:
#             result += " "
#         result += "(" + str(params[i]) + " " + str(params[i].getSort()) + ")"
#     result += ") " + str(sort) + " " + str(body) + ")"
#     return result
#
# def print_synth_solutions(terms, sols):
#     result = ""
#     for i in range(0, len(terms)):
#         params = []
#         if sols[i].getKind() == Kind.Lambda:
#             params += sols[i][0]
#             body = sols[i][1]
#         result += str(body)
#     print(result)
from sygus.sygus_to_asp import SyGuSToASP


def unsat_core_by_fun(unsat_core: Iterable[Symbol]) -> Dict[str, List[Symbol]]:
    res = defaultdict(list)

    for sym in unsat_core:
        if sym.type == SymbolType.Function:
            res[sym.name].append(sym)
        else:
            raise NotImplementedError()

    return res


class SyGuSVisitor:

    def __init__(self, instance: Instance, unsat_cores: Collection[Collection[Symbol]], valid_models,
                 relax_pbe_constraints=False, constrain_reflexive=False, skip_cores=0):
        self.instance = instance
        self.unsat_cores = unsat_cores
        self.models = valid_models
        self.relax_pbe_constraints = relax_pbe_constraints

        self.solver = cvc5.Solver()

        self.solver.setOption('sygus', 'true')
        self.solver.setOption('lang', 'sygus2')
        self.solver.setOption('incremental', 'true')
        self.solver.setOption('sygus-enum', 'fast')

        self.solver.setLogic('ALL')

        Int = self.solver.getIntegerSort()
        Bool = self.solver.getBooleanSort()

        min_core = sorted(unsat_cores, key=len)[skip_cores:][0]

        self.vars_by_sym = defaultdict(lambda: defaultdict(list))
        self.sym_by_vars = {}
        self.num_by_sym = defaultdict(lambda: 0)

        self.sygus = SyGuSProblem('f')
        if self.instance.constants:
            self.sygus.make_enum(self.solver, self.instance.constants + ['empty'])
        else:
            self.sygus.make_enum(self.solver, ['empty'])

        for sym_i, sym in enumerate(min_core):
            self.num_by_sym[sym.name] += 1

            if sym.type == SymbolType.Function:
                for arg_i, arg in enumerate(sym.arguments):
                    if arg.type == SymbolType.Number:
                        self.sygus.add_input(Int, self.solver.mkVar(Int,
                                                                    f'{sym.name}_{string.ascii_uppercase[sym_i]}_{arg_i}'))
                        self.vars_by_sym[sym.name][sym_i].append(self.sygus.inputs[-1])
                        self.sym_by_vars[self.sygus.inputs[-1]] = (sym.name, sym_i, arg_i)
                    elif arg.type == SymbolType.Function and len(arg.arguments) == 0:
                        self.sygus.add_input(self.sygus.constantSort, self.solver.mkVar(self.sygus.constantSort,
                                                                                        f'{sym.name}_{string.ascii_uppercase[sym_i]}_{arg_i}'))
                        self.vars_by_sym[sym.name][sym_i].append(self.sygus.inputs[-1])
                        self.sym_by_vars[self.sygus.inputs[-1]] = (sym.name, sym_i, arg_i)
                        pass
                    else:
                        raise NotImplementedError()
            else:
                raise NotImplementedError()

        S = self.solver.mkVar(Bool, 'S')
        B = self.solver.mkVar(Bool, 'B')
        I = self.solver.mkVar(Int, 'I')
        C = self.solver.mkVar(self.sygus.constantSort, 'C')

        sortMapper = {Int: I, Bool: B, self.sygus.constantSort: C}

        self.sygus.set_grammar_rules({
            S: {B,
                (Kind.OR, B, S)},
            B: {(Kind.NOT, B),
                (Kind.AND, B, B),
                (Kind.EQUAL, I, I), (Kind.LT, I, I), (Kind.GT, I, I),
                (Kind.EQUAL, C, C)},
            I: {(Kind.ADD, I, I),
                (Kind.SUB, I, I),
                (Kind.ABS, I),
                0, 1, 2},
            C: [(Kind.APPLY_CONSTRUCTOR, self.sygus.constantSort.getDatatype().getConstructor(const).getTerm()) for
                const in self.sygus.constructors.keys()]
        }, self.solver)

        for sort, vars in self.sygus.inputs_by_sort.items():
            for var in vars:
                self.sygus.add_grammar_rule(sortMapper[sort], var)

        self.sygus.set_starting_symbol(S)

        self.f = self.sygus.get_synth_fun(self.solver)

        if constrain_reflexive:
            for sym in self.vars_by_sym.keys():
                temp_vars = [self.solver.declareSygusVar(string.ascii_lowercase[var_i], var.getSort()) for var_i, var in
                             enumerate(self.sygus.inputs)]

                args_permutations = list((map(list, map(itertools.chain.from_iterable,
                                                        itertools.permutations(self.vars_by_sym[sym].values())))))
                args_permutations = [[temp_vars[self.sygus.inputs.index(elem)] for elem in perm] for perm in
                                     args_permutations]

                if args_permutations:
                    for args1, args2 in zip(itertools.repeat(temp_vars), args_permutations):
                        term1 = self.solver.mkTerm(Kind.APPLY_UF, self.f, *args1)
                        term2 = self.solver.mkTerm(Kind.APPLY_UF, self.f, *args2)
                        self.sygus.add_constraint(self.solver.mkTerm(Kind.EQUAL, term1, term2))

    def solve(self, max_cores, max_models, skip_cores=0):
        # print(self.sygus)
        print('Using the following UNSAT cores as positive examples:')
        for unsat_core in sorted(self.unsat_cores, key=len)[skip_cores:max_cores]:
            print(list(map(str, unsat_core)))

            constraints = []
            for combo in self.get_combos(unsat_core):
                args = self.combo_to_args(combo)
                constraint = self.solver.mkTerm(Kind.APPLY_UF, self.f, *args)
                constraints.append(constraint)

            if len(constraints) > 1:
                mk_term = self.solver.mkTerm(Kind.OR, *constraints)
                # print(mk_term)
                self.sygus.add_constraint(mk_term)
            else:
                self.sygus.add_constraint(constraints[0])

        print()
        if len(self.models) >= 1:
            print('Using the following valid models as negative examples:')
            for model in sorted(self.models, key=len)[:max_models]:
                print(model)

                for combo in self.get_combos(model):
                    args = self.combo_to_args(combo)
                    # print(args)
                    constraint = self.solver.mkTerm(Kind.NOT, self.solver.mkTerm(Kind.APPLY_UF, self.f, *args))
                    self.sygus.add_constraint(constraint)

        else:
            variables = [self.solver.mkVar(var.getSort()) for var in self.sygus.inputs]
            variable_list = self.solver.mkTerm(Kind.VARIABLE_LIST, *variables)
            constraint = self.solver.mkTerm(Kind.APPLY_UF, self.f, *variables)
            constraint = self.solver.mkTerm(Kind.NOT, constraint)
            constraint = self.solver.mkTerm(Kind.EXISTS, variable_list, constraint)

            self.sygus.add_constraint(constraint)

        print()
        print('SyGuS Spec:')
        print(self.sygus)

        self.sygus.realize_constraints(self.solver)

        # print(self.sygus)

        print('\nSuggestions:\n')

        sygus_to_asp = SyGuSToASP(self.sym_by_vars, self.vars_by_sym)

        for solution in self.sygus.enumerate(self.solver):
            logger.info('Solution Found')
            print(solution)
            print(sygus_to_asp.visit_statement(solution))
            print()

    def combo_to_args(self, combo):
        args = []
        for sym in combo:
            for arg in sym.arguments:
                if arg.type == SymbolType.Number:
                    args.append(self.solver.mkInteger(arg.number))
                elif arg.type == SymbolType.Function and len(arg.arguments) == 0:
                    args.append(self.solver.mkTerm(Kind.APPLY_CONSTRUCTOR,
                                                   self.sygus.constantSort.getDatatype().getConstructor(
                                                       arg.name).getTerm()))
                else:
                    raise NotImplementedError()

        return args

    def get_combos(self, unsat_core):
        unsat_core_dict = unsat_core_by_fun(unsat_core)
        product_parts = []
        for sym_name, n in self.num_by_sym.items():
            if not self.relax_pbe_constraints:
                product_parts.append(list(itertools.combinations(unsat_core_dict[sym_name], n)))
            else:
                product_parts.append(list(itertools.permutations(unsat_core_dict[sym_name], n)))
        combos = list(map(list, map(itertools.chain.from_iterable, itertools.product(*product_parts))))
        return combos
