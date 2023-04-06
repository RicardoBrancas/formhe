#!/usr/bin/env python

import logging
import traceback
from itertools import chain

import clingo.ast

import runhelper
from asp.highlithing_visitor import AspHighlithingVisitor, RuleScoreCalculator, GeometricScorer, ArithmeticScorer, MaxScorer
from formhe.exceptions.parser_exceptions import InstanceParseException, InstanceGroundingException
from ordered_set import OrderedSet

from formhe.asp.instance import Instance
from formhe.asp.synthesis.ASPSpecGenerator import ASPSpecGenerator
from formhe.asp.synthesis.AspInterpreter import AspInterpreter
from formhe.asp.synthesis.AspVisitor import AspVisitor
from formhe.asp.synthesis.StatementEnumerator import StatementEnumerator
from formhe.trinity.z3_enumerator import Z3Enumerator
from formhe.utils import config
from formhe.utils import perf

logger = logging.getLogger('formhe.asp.integrated')


def main():
    logger.info('Starting FormHe ASP bug finder')
    logger.info('%s', config.get())

    logger.info('Loading instance from %s', config.get().input_file)
    try:
        instance = Instance(config.get().input_file)
    except InstanceParseException:
        print("Error while parsing input file.")
        exit(-1)
    except InstanceGroundingException:
        print("Error while grounding.")
        exit(-1)

    problems_found = False

    for i in range(len(instance.instrumented_asts)):
        runhelper.timer_start('answer.set.enum.time')
        instance.compute_models(0, i)
        instance.ground_truth.compute_models(0, i)
        runhelper.timer_stop('answer.set.enum.time')

        if len(instance.models[i]) == len(instance.ground_truth.models[i]) == 0:
            continue

        if len(instance.models[i]) == 0:
            logger.info('Your solution is overconstrained and does not produce any solutions for input %d.', i)
            problems_found = True
            continue

        if instance.models[i] <= instance.ground_truth.models[i]:
            continue

        else:
            logger.info('Your solution is underconstrained and produces the following wrong models for input %d:', i)

            for model in instance.models[i] - instance.ground_truth.models[i]:
                logger.info(' '.join(map(str, model)))

            problems_found = True
            continue

    if not problems_found:
        print('No problems found')
        exit()

    missing_models = list(chain.from_iterable(instance.missing_models))

    print(missing_models)

    highlighter = AspHighlithingVisitor()

    highlighter.highlight(config.get().input_file, missing_models)

    geometric_scorer = MaxScorer()
    rule_score_calculator = RuleScoreCalculator(highlighter.graph.weights, geometric_scorer)

    clingo.ast.parse_files([config.get().input_file], lambda x: rule_score_calculator.visit(x))

    for rule, score in rule_score_calculator.scorer.get_scores():
        print(rule, score)

    runhelper.log_any('scores', rule_score_calculator.scorer.get_scores())

    if instance.config.selfeval_lines is not None:
        faulty_lines = set(instance.config.selfeval_lines)
        runhelper.log_any('selfeval.lines', faulty_lines)
        selected_lines = set([rule for rule, score in sorted(rule_score_calculator.scorer.get_scores(), key=lambda x: x[1], reverse=True)[:1]])
        runhelper.log_any('selected.lines', selected_lines)
        if not faulty_lines and not selected_lines:
            runhelper.log_any('fault.identified', 'Yes (no incorrect lines)')
        elif not faulty_lines:
            runhelper.log_any('fault.identified', 'No (no incorrect lines)')
        elif faulty_lines == selected_lines:
            runhelper.log_any('fault.identified', 'Yes')
        elif faulty_lines < selected_lines:
            runhelper.log_any('fault.identified', 'Superset')
        elif faulty_lines.intersection(selected_lines):
            runhelper.log_any('fault.identified', 'Subset')
        else:
            if not selected_lines:
                runhelper.log_any('fault.identified', 'No (no lines identified)')
            else:
                runhelper.log_any('fault.identified', 'No (wrong lines identified)')

        if len(instance.config.selfeval_lines) > 0 and instance.config.selfeval_fix_test and instance.config.selfeval_fix is not None:
            spec_generator = ASPSpecGenerator(instance.ground_truth, 0, instance.constantCollector.predicates.items())
            trinity_spec = spec_generator.trinity_spec
            asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)
            asp_interpreter = AspInterpreter(instance, instance.constantCollector.predicates.keys())

            if asp_interpreter.test(instance.config.selfeval_fix):
                runhelper.log_any('fault.partial', 'Yes')
            else:
                runhelper.log_any('fault.partial', 'No')


if __name__ == '__main__':
    main()
