import logging
from collections import Counter, defaultdict
from typing import List

import clingo
from clingo.ast import ASTType
from ordered_set import OrderedSet

import runhelper
from fl.fault_localizer import FaultLocalizer
from formhe.asp.instance import Instance
from utils import print_utils
from utils.dc import combine_improved

logger = logging.getLogger('formhe.asp.fault_localization')


def mcs_negated_query(model, relaxed):
    if not relaxed:
        negated_query = ' '.join([':- not ' + str(x) + '.' for x in model if x])
    else:
        negated_query = ' '.join(['' + str(x) + '.' for x in model if x])
    return negated_query


def mcs_query(model):
    query = ':- ' + (', '.join([str(x) for x in model if x])) + '.'
    return query


def all_mcs(instance, model, relaxed=False, i=0, positive=False):
    # logger.info(f'Iterating all {"relaxed " if relaxed else ""}MCSs')

    instrumenter_vars = instance.instrumenter.relaxation_functions
    generate_vars = '0 { ' + '; '.join(map(str, instrumenter_vars)) + ' } ' + str(len(instrumenter_vars)) + '.'
    clause_r = ''
    mcs_blocks = OrderedSet()
    mcss = OrderedSet()

    if positive:
        negated_query = ' '.join([mcs_query(m) for m in model])
    else:
        negated_query = mcs_negated_query(model, relaxed)

    # logger.info('Transformed query: %s', negated_query)

    unsatisfied_vars = None
    unsat = True
    while True:
        ctl = instance.get_control(generate_vars, clause_r, negated_query, *mcs_blocks, max_sols=1, instrumented=True, i=i)

        with ctl.solve(yield_=True) as handle:
            res = handle.get()

            if res.satisfiable:
                unsat = False
                m = handle.model()
                unsatisfied_vars = []
                satisfied_vars = []

                for var in instrumenter_vars:
                    if not m.contains(var):
                        unsatisfied_vars.append(var)
                    else:
                        satisfied_vars.append(var)

                unsatisfied_vars = frozenset(unsatisfied_vars)
                satisfied_vars = frozenset(satisfied_vars)

                # print(clause_r if clause_r else generate_vars)
                # for mcs_block in mcs_blocks:
                #     print(mcs_block)
                # print('SAT')
                # print('satisfied:', list(map(str, satisfied_vars)))
                # print('unsatisfied:', list(map(str, unsatisfied_vars)))
                # print('\n\n\n')

                clause_r = f'{len(satisfied_vars) + 1} {{ {"; ".join(map(str, instrumenter_vars))} }} {len(instrumenter_vars)}.'

            elif res.unsatisfiable:
                # print(clause_r if clause_r else generate_vars)
                # for mcs_block in mcs_blocks:
                #     print(mcs_block)
                # print('UNSAT')
                # print('\n\n\n')

                if not unsat:
                    if not any(map(lambda mcs: mcs.issubset(unsatisfied_vars), mcss)):
                        yield frozenset(instance.instrumenter.relaxations_function_map[var] for var in unsatisfied_vars)
                        mcss.append(unsatisfied_vars)
                        # logger.info(' '.join(str(x) for x in unsatisfied_vars))
                        clause_r = ''
                        mcs_blocks.append(':- ' + ','.join(map(lambda x: 'not ' + str(x), unsatisfied_vars)) + '.')
                        unsat = True
                    else:
                        raise NotImplementedError()
                        # clause_r = ''
                        # mcs_blocks.append(':- ' + ','.join(map(lambda x: 'not ' + str(x), unsatisfied_vars)) + '.')
                        # unsat = True
                else:
                    return

            else:
                raise RuntimeError()


def all_mfl(instance, model, relaxed=False, i=0, positive=False):
    ctl = instance.get_control(max_sols=1, i=i)

    supports = []
    support_vars = []

    for symbolic_atom in ctl.symbolic_atoms:
        symbol = symbolic_atom.symbol
        supports.append(f'{symbol} :- not _support({symbol}).')
        support_vars.append(clingo.Function(f"_support", [symbol]))

    instrumenter_vars = instance.instrumenter.relaxation_functions
    n = len(instrumenter_vars + support_vars)
    generate_vars = '0 { ' + '; '.join(map(str, instrumenter_vars + support_vars)) + ' } ' + str(n) + '.'
    clause_r = ''
    mcs_blocks = OrderedSet()
    mcss = OrderedSet()

    if positive:
        negated_query = ' '.join([mcs_query(m) for m in model])
    else:
        negated_query = mcs_negated_query(model, relaxed)

    # logger.info('Transformed query: %s', negated_query)

    unsatisfied_vars = None
    unsat = True
    while True:
        ctl = instance.get_control(generate_vars, clause_r, negated_query, ' '.join(supports), *mcs_blocks, max_sols=1, instrumented=True, i=i)

        with ctl.solve(yield_=True) as handle:
            res = handle.get()

            if res.satisfiable:
                unsat = False
                m = handle.model()
                unsatisfied_vars = []
                satisfied_vars = []

                for var in instrumenter_vars + support_vars:
                    if not m.contains(var):
                        unsatisfied_vars.append(var)
                    else:
                        satisfied_vars.append(var)

                unsatisfied_vars = frozenset(unsatisfied_vars)
                satisfied_vars = frozenset(satisfied_vars)

                # print(clause_r if clause_r else generate_vars)
                # for mcs_block in mcs_blocks:
                #     print(mcs_block)
                # print('SAT')
                # print('satisfied:', list(map(str, satisfied_vars)))
                # print('unsatisfied:', list(map(str, unsatisfied_vars)))
                # print('\n\n\n')

                clause_r = f'{len(satisfied_vars) + 1} {{ {"; ".join(map(str, instrumenter_vars + support_vars))} }} {n}.'

            elif res.unsatisfiable:
                # print(clause_r if clause_r else generate_vars)
                # for mcs_block in mcs_blocks:
                #     print(mcs_block)
                # print('UNSAT')
                # print('\n\n\n')

                if not unsat:
                    if not any(map(lambda mcs: mcs.issubset(unsatisfied_vars), mcss)):
                        unsatisfied_rules = [instance.instrumenter.relaxations_function_map[var] for var in unsatisfied_vars if var in instrumenter_vars]
                        for var in unsatisfied_vars:
                            if var in support_vars:
                                predicate = var.arguments[0].name
                                for rule_i, rule in enumerate(instance.instrumenter.rules):
                                    if rule.ast_type == ASTType.Rule and \
                                            rule.head and \
                                            rule.head.ast_type == ASTType.Literal and \
                                            rule.head.atom.ast_type == ASTType.SymbolicAtom and \
                                            rule.head.atom.symbol.name == predicate:
                                        unsatisfied_rules.append(rule_i)
                                    elif rule.ast_type == ASTType.Rule and \
                                            rule.head and \
                                            rule.head.ast_type == ASTType.Aggregate:
                                        for element in rule.head.elements:
                                            if element.ast_type == ASTType.ConditionalLiteral and \
                                                    element.literal.ast_type == ASTType.Literal and \
                                                    element.literal.atom.ast_type == ASTType.SymbolicAtom and \
                                                    element.literal.atom.symbol.name == predicate:
                                                unsatisfied_rules.append(rule_i)
                        yield frozenset(unsatisfied_rules)
                        mcss.append(unsatisfied_vars)
                        # logger.info(' '.join(str(x) for x in unsatisfied_vars))
                        clause_r = ''
                        mcs_blocks.append(':- ' + ','.join(map(lambda x: 'not ' + str(x), unsatisfied_vars)) + '.')
                        unsat = True
                    else:
                        raise NotImplementedError()
                        # clause_r = ''
                        # mcs_blocks.append(':- ' + ','.join(map(lambda x: 'not ' + str(x), unsatisfied_vars)) + '.')
                        # unsat = True
                else:
                    return

            else:
                raise RuntimeError()


def union(*args):
    if len(args) > 0:
        base = args[0]
        for other in args[1:]:
            base = base.union(other)
        return base
    else:
        return []


class MCSFL(FaultLocalizer):

    def __init__(self, instance: Instance):
        super().__init__(instance)
        self.mcs_hit_counter = Counter()
        self.mcss_negative_non_relaxed = defaultdict(lambda: OrderedSet())
        self.mcss_negative_relaxed = defaultdict(lambda: OrderedSet())
        self.mfl = defaultdict(lambda: OrderedSet())
        self.mcss_positive = defaultdict(lambda: OrderedSet())

    def fault_localize(self) -> List:
        runhelper.log_any('models.missing', [len(x) for x in self.instance.missing_models])
        runhelper.log_any('models.extra', [len(x) for x in self.instance.extra_models])

        for i in range(len(self.instance.asts)):
            for model in self.instance.missing_models[i]:
                # if instance.missing_models[i]:
                #     model = instance.missing_models[i][0]
                runhelper.timer_start('fl.mcs.time')
                if not self.instance.config.skip_mcs_negative_non_relaxed:
                    for mcs in all_mcs(self.instance, model, relaxed=False, i=i):
                        self.mcss_negative_non_relaxed[i].append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mcs.time')
                runhelper.timer_start('fl.mcs.time')
                if not self.instance.config.skip_mcs_negative_relaxed:
                    for mcs in all_mcs(self.instance, model, relaxed=True, i=i):
                        self.mcss_negative_relaxed[i].append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mcs.time')
                runhelper.timer_start('fl.mfl.time')
                if self.instance.config.use_mfl:
                    for mcs in all_mfl(self.instance, model, relaxed=False, i=i):
                        self.mfl[i].append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mfl.time')

            if self.instance.extra_models[i]:
                runhelper.timer_start('fl.mcs.time')
                if self.instance.config.use_mcs_positive or ((not self.instance.config.skip_mcs_positive_conditional) and all(map(lambda x: len(x) == 0, self.instance.missing_models))):
                    for mcs in all_mcs(self.instance, self.instance.extra_models[i], i=i, positive=True):
                        self.mcss_positive[i].append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mcs.time')

        runhelper.log_any('symbolic.atoms.first', len(self.instance.controls[0].symbolic_atoms))
        runhelper.log_any('symbolic.atoms.all', [len(ctl.symbolic_atoms) for ctl in self.instance.controls])
        runhelper.log_any('fl.mcss.hit.count', self.mcs_hit_counter)

        runhelper.log_any('fl.mcss.negative.non.relaxed', print_utils.simplify(self.mcss_negative_non_relaxed))
        runhelper.log_any('fl.mcss.negative.relaxed', print_utils.simplify(self.mcss_negative_relaxed))
        runhelper.log_any('fl.mcss.positive', print_utils.simplify(self.mcss_positive))
        runhelper.log_any('fl.mfl', print_utils.simplify(self.mfl))

        diags = []

        for i in range(len(self.instance.asts)):
            diags_for_i = OrderedSet()

            diags_for_i.update(self.mcss_negative_relaxed[i])
            diags_for_i.update(self.mcss_negative_non_relaxed[i])
            diags_for_i.update(self.mcss_positive[i])
            diags_for_i.update(self.mfl[i])

            tmp = OrderedSet([d for d in diags_for_i if len(d) > 0])
            if len(tmp) > 0:
                diags.append(tmp)

        runhelper.log_any("diags", print_utils.simplify(diags))
        runhelper.timer_start("diags.combination")
        diags_combined = combine_improved(diags)
        runhelper.timer_stop("diags.combination")
        runhelper.log_any("diags.combined", print_utils.simplify(diags_combined))
        runhelper.log_any("diags.different", set(diags_combined) != set([tuple(diag) for diag in union(*diags)]))

        # runhelper.log_any('fl.all', [set(fl) for fl in fls])

        return [frozenset(d) for d in diags_combined]
