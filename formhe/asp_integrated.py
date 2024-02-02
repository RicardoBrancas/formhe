#!/usr/bin/env python
import logging
import math
import re
import traceback
from collections import Counter
from itertools import chain
import random

import clingo.ast

from fl.Combination import CombinationFaultLocalizer
from fl.Debug import DebugFL
from formhe.asp.highlithing_visitor import AspHighlithingVisitor, MaxScorer, RuleScoreCalculator
from ordered_set import OrderedSet

import runhelper
from formhe.asp.instance import Instance
from formhe.asp.synthesis.ASPSpecGenerator import ASPSpecGenerator
from formhe.asp.synthesis.AspInterpreter import AspInterpreter
from formhe.asp.synthesis.AspVisitor import AspVisitor
from formhe.asp.synthesis.StatementEnumerator import StatementEnumerator
from formhe.exceptions.parser_exceptions import InstanceParseException, InstanceGroundingException
from formhe.trinity.z3_enumerator import Z3Enumerator
from formhe.utils import config

logger = logging.getLogger('formhe.asp.integrated')


def main():
    logger.info('Starting FormHe ASP bug finder')
    logger.info('%s', config.get())
    runhelper.init_logger("formhe.runhelper")
    runhelper.register_sigterm_handler()

    runhelper.timer_start('fault.localization.time')

    logger.info('Loading instance from %s', config.get().input_file)
    try:
        instance = Instance(config.get().input_file)
    except InstanceParseException:
        print("Error while parsing input file.")
        exit(-1)
    except InstanceGroundingException:
        print("Error while grounding.")
        exit(-1)

    for i, instrumented_ast in enumerate(instance.instrumented_asts):
        logger.debug('Instrumented program %d:\n%s', i, '\n'.join(str(x) for x in instrumented_ast))

    problems_found = False

    already_printed = False
    passed = 0
    failed = 0
    for i in range(len(instance.instrumented_asts)):
        runhelper.timer_start('answer.set.enum.time')
        instance.compute_models(0, i)
        instance.ground_truth.compute_models(0, i)
        runhelper.timer_stop('answer.set.enum.time')

        if len(instance.models[i]) == len(instance.ground_truth.models[i]) == 0:
            passed += 1
            continue

        if len(instance.models[i]) == 0 or (len(instance.models[i]) == 1 and len(instance.models[i][0]) == 0 and (len(instance.ground_truth.models[i]) != 1 or len(instance.ground_truth.models[i][0]) != 0)):
            logger.info('Your solution is overconstrained and does not produce any solutions for input %d.', i)

            if not already_printed or not config.get().print_only_first_test_case:
                print(f"Your program has failed test case {i}:")
                print()
                print(re.sub('^', '\t', instance.config.instance_base64[i], flags=re.MULTILINE))
                print()
                print(f'Your solution is overconstrained and does not produce any solutions for this input. Examples of correct answer sets:')
                print()
                for m in instance.ground_truth.models[i][:5]:
                    print('\t' + ' '.join(map(str, m)))

                print()

            already_printed = True
            problems_found = True
            failed += 1
            continue

        if instance.models[i] <= instance.ground_truth.models[i]:
            passed += 1
            continue

        else:
            logger.info('Your solution is underconstrained and produces the following wrong models for input %d:', i)

            if not already_printed or not config.get().print_only_first_test_case:
                print(f"Your program has failed test case {i}:")
                print()
                print(re.sub('^', '\t', instance.config.instance_base64[i], flags=re.MULTILINE))
                print()

                print(f'Your solution is underconstrained and produces the following wrong models for this input:')
                print()

            # logger.info('(models omitted for performance reasons)')
            for model in instance.models[i] - instance.ground_truth.models[i]:
                if model:
                    if not already_printed or not config.get().print_only_first_test_case:
                        print('\t' + ' '.join(map(str, model)))
                    logger.info(' '.join(map(str, model)))
                else:
                    if not already_printed or not config.get().print_only_first_test_case:
                        print('\t<empty answer set>')
                    logger.info('<empty answer set>')

            if not already_printed or not config.get().print_only_first_test_case:
                print()
            already_printed = True
            problems_found = True
            failed += 1
            continue

    print(f'You have [+passed {passed} test(s)+] and [-failed {failed} test(s)-].\n')

    if not problems_found:
        print('No problems found\n')
        exit()

    fault_localizer = CombinationFaultLocalizer(instance)
    mcss = fault_localizer.fault_localize()

    if instance.config.simulate_fault_localization and instance.config.selfeval_lines is not None:
        debug_fl = DebugFL(instance)
        mcss = debug_fl.fault_localize()

    predicates_before_skip = instance.constantCollector.predicates

    runhelper.log_any('predicates.unsupported', str(set(OrderedSet(predicates_before_skip.keys()) - instance.constantCollector.predicates_generated)))
    runhelper.log_any('predicates.needed', str(set(instance.constantCollector.predicates_used - instance.constantCollector.predicates_generated)))

    if set(instance.constantCollector.predicates_used - instance.constantCollector.predicates_generated):
        print('The following predicates are used in a rule but are never generated: ' + ' '.join(map(str, set(instance.constantCollector.predicates_used - instance.constantCollector.predicates_generated))))
        print()

    if mcss[0]:

        print()
        print('Suggested lines where bug is likely located:  ')

        for line in mcss[0]:
            if not line:
                continue
            print(f'**[{line}]**', '`' + instance.instrumenter.original_rules[line] + '`', '  ')

        print()
        print()


    runhelper.timer_stop('fault.localization.time')
    if instance.config.exit_after_fault_localization:
        for i in range(len(instance.asts)):
            instance.get_control(i=i)
        exit()

    # for a in instance.constantCollector.skipped:
    #     print(a)

    # print(preset_atoms[-1])

    # perf.timer_start(perf.ANSWER_SET_ENUM_TIME)
    # if not config.get().no_semantic_constraints:
    #     instance.find_wrong_models(max_sols=1000)
    # instance.generate_correct_models(config.get().n_gt_sols_generated)
    # perf.timer_stop(perf.ANSWER_SET_ENUM_TIME)
    #
    # sorted_cores = sorted(instance.cores, key=len)

    # runhelper.timer_start('answer.set.enum.time')
    # for i in range(len(instance.instrumented_asts)):
    #     instance.ground_truth.compute_models(0, i)  # todo this has been done before already
    # runhelper.timer_stop('answer.set.enum.time')

    solved = False
    first_depth = True
    for depth in range(config.get().minimum_depth, config.get().maximum_depth):
        for extra_statements in [[], [('empty', [])], [('empty', [None])], [('empty', [None]), ('empty', [None])]]:
            for mcs in mcss:
                modified_instance = Instance(config.get().input_file, skips=mcs, ground_truth_instance=instance.ground_truth)
                spec_generator = ASPSpecGenerator(modified_instance, modified_instance.config.extra_vars, predicates_before_skip.items())
                trinity_spec = spec_generator.trinity_spec
                asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)
                preset_statements = []
                for rule in modified_instance.constantCollector.skipped:
                    preset_statements.append(asp_visitor.visit(rule))
                asp_interpreter = AspInterpreter(modified_instance, predicates_before_skip.keys())

                # runhelper.timer_start('answer.set.enum.time')
                # runhelper.timer_start('answer.set.enum.time.2')
                # if not config.get().no_semantic_constraints:
                #     modified_instance.find_wrong_models(max_sols=1000)
                # runhelper.timer_stop('answer.set.enum.time')
                # runhelper.timer_stop('answer.set.enum.time.2')

                # sorted_cores = sorted(modified_instance.cores, key=len)

                atom_enum_constructor = lambda p: Z3Enumerator(trinity_spec, depth, predicates_names=modified_instance.constantCollector.predicates.keys(), cores=None, free_vars=asp_visitor.free_vars,
                                                               preset_statements=list(p), strict_minimum_depth=not first_depth,
                                                               free_predicates=OrderedSet(predicates_before_skip.keys()) - modified_instance.constantCollector.predicates_generated,
                                                               force_generate_predicates=modified_instance.constantCollector.predicates_used - modified_instance.constantCollector.predicates_generated)
                statement_enumerator = StatementEnumerator(atom_enum_constructor, preset_statements + extra_statements, 1, asp_visitor.free_vars, depth)

                logger.info('Unsupported predicates: %s', str(set(OrderedSet(predicates_before_skip.keys()) - modified_instance.constantCollector.predicates_generated)))
                logger.info('Needed predicates: %s', str(set(modified_instance.constantCollector.predicates_used - modified_instance.constantCollector.predicates_generated)))

                runhelper.timer_start('eval.fail.time')
                while prog := next(statement_enumerator):
                    runhelper.tag_increment('enum.programs')
                    runhelper.timer_start('eval.time')
                    try:
                        asp_prog = asp_interpreter.eval(prog)
                        # print(asp_prog)
                        # logger.debug(prog)
                        # logger.debug(asp_prog)

                        res = asp_interpreter.test(asp_prog)
                        if res:
                            logger.info('Solution found')

                            print('**Fix Suggestion**\n')

                            if mcs:
                                print(f'You can try replacing the following line{"s" if len(statement_enumerator.current_preset_statements) > 1 else ""}:\n')
                                print('\n'.join(['\t' + str(line) for line in mcs]))
                                print('\nWith (the "?" are missing parts you should fill in):\n')
                                print('\n'.join(['\t' + str(line) for line in statement_enumerator.current_preset_statements]))
                                print()
                            else:
                                print(f'You can try adding the following line{"s" if len(statement_enumerator.current_preset_statements) > 1 else ""} (the "?" are missing parts you should fill in):\n')
                                print('\n'.join(['\t' + str(line) for line in statement_enumerator.current_preset_statements]))
                                print()

                            # print('Solution found')
                            # print(mcs)
                            # print(statement_enumerator.current_preset_statements)
                            # print(asp_prog)
                            solved = True
                            break
                    except (RuntimeError, InstanceGroundingException) as e:
                        runhelper.timer_stop('eval.fail.time')
                        runhelper.tag_increment('eval.fail.programs')
                        logger.warning('Failed to parse: %s', prog)
                        # traceback.print_exception(e)
                        # exit()
                    except Exception as e:
                        # traceback.print_exception(e)
                        raise e
                    runhelper.timer_stop('eval.time')
                    runhelper.timer_start('eval.fail.time')

                if solved:
                    break
            if solved:
                break
        if solved:
            break

        first_depth = False

    if not solved:
        print('Synthesis Failed')
        exit(-2)


if __name__ == '__main__':
    main()
