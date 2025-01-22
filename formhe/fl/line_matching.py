import logging
import traceback
from itertools import product
from typing import List

import clingo.ast
import numpy as np
from multiset import Multiset
from ordered_set import OrderedSet
from scipy.optimize import linear_sum_assignment

import runhelper
from asp.instance import Instance
from formhe.fl.fault_localizer import FaultLocalizer

logger = logging.getLogger('formhe.asp.fault_localization')


def bag_of_nodes(instance):
    from formhe.asp.synthesis.AspVisitor import AspVisitor, bag_nodes
    from formhe.asp.synthesis.spec_generator import ASPSpecGenerator

    spec_generator = ASPSpecGenerator(instance, 0, instance.predicates.items())
    asp_visitor = AspVisitor(spec_generator.trinity_spec, spec_generator.free_vars, True, domain_predicates=instance.problem.output_predicates_tuple + instance.problem.input_predicates_tuple)

    rules_bags = []
    for rule in instance.base_ast:
        if rule.ast_type != clingo.ast.ASTType.Rule and rule.ast_type != clingo.ast.ASTType.Definition:
            continue
        node = asp_visitor.visit(rule)
        rules_bags.append(bag_nodes(node))

    return rules_bags


def var_anonymized_rules(instance):
    from formhe.asp.synthesis.AspVisitor import AspVisitor
    from formhe.asp.synthesis.spec_generator import ASPSpecGenerator

    spec_generator = ASPSpecGenerator(instance, 0, instance.predicates.items())
    asp_visitor = AspVisitor(spec_generator.trinity_spec, spec_generator.free_vars, anonymize_vars=True, domain_predicates=instance.problem.output_predicates_tuple + instance.problem.input_predicates_tuple)

    anonymized_rules = []
    for rule in instance.base_ast:
        if rule.ast_type != clingo.ast.ASTType.Rule and rule.ast_type != clingo.ast.ASTType.Definition:
            continue
        node = asp_visitor.visit(rule)
        anonymized_rules.append(node)

    return anonymized_rules


def pairings_with_cost(instance, other_instance=None, compute_fully_matching_rules=False):
    if other_instance is None:
        other_instance = instance.reference

    instance_bags = bag_of_nodes(instance)
    other_bags = bag_of_nodes(other_instance)

    instance_anon_rules = var_anonymized_rules(instance)
    other_anon_rules = var_anonymized_rules(other_instance)

    if len(other_bags) < len(instance_bags):
        n = len(instance_bags) - len(other_bags)
        other_bags += [Multiset(['empty']) for _ in range(n)]
        other_anon_rules += [None for _ in range(n)]

    if len(instance_bags) < len(other_bags):
        n = len(other_bags) - len(instance_bags)
        instance_bags += [Multiset(['empty']) for _ in range(n)]
        instance_anon_rules += [None for _ in range(n)]

    costs = np.zeros((len(other_bags), len(instance_bags)))
    for (i, a), (j, b) in product(enumerate(other_bags), enumerate(instance_bags)):
        costs[i, j] = len(a.symmetric_difference(b))
        if costs[i, j] == 0 and other_anon_rules[i] is not None and instance_anon_rules[j] is not None and not other_anon_rules[i].deep_eq(instance_anon_rules[j]):
            costs[i, j] = 0.01

    row_ind, col_ind = linear_sum_assignment(costs)

    pairings_with_cost = []
    fully_matching_rules = []
    for a, b in zip(row_ind, col_ind):
        if compute_fully_matching_rules and other_anon_rules[a] is not None and instance_anon_rules[b] is not None and other_anon_rules[a].deep_eq(instance_anon_rules[b]):
            fully_matching_rules.append(b)

        pairings_with_cost.append((a, b, costs[a, b]))

    if not compute_fully_matching_rules:
        return pairings_with_cost
    else:
        return pairings_with_cost, fully_matching_rules


class LineMatchingFL(FaultLocalizer):

    def __init__(self, instance: Instance):
        super().__init__(instance)
        self.fully_matching_rules = []

    def fault_localize(self) -> List:
        runhelper.timer_start('fl.line.matching.time')
        try:
            pwc, fmr = pairings_with_cost(self.instance, compute_fully_matching_rules=True)

            runhelper.log_any('fl.pairing.line.scores', [(b, s) for a, b, s in pwc])
            runhelper.log_any('fl.pairing.matching.rules', fmr)

            self.fully_matching_rules = fmr

            mcs = OrderedSet()
            if pwc:
                for a, b, cost in pwc:
                    if cost <= self.instance.config.line_matching_threshold and cost != 0 and cost != 0.01:
                        mcs.add(b)

            runhelper.timer_stop('fl.line.matching.time')
            return [tuple(mcs)]

        except Exception as e:
            traceback.print_exception(e)
            print(e)
            logger.error('Exception while trying to compute line pairings')
            runhelper.timer_stop('fl.line.matching.time')
            return [tuple()]
