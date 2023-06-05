#!/usr/bin/env python
import logging
import traceback
from collections import Counter
from itertools import chain

import clingo.ast

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

    logger.info('Loading instance from %s', config.get().input_file)
    try:
        instance = Instance(config.get().input_file)
    except InstanceParseException:
        print("Error while parsing input file.")
        exit(-1)
    except InstanceGroundingException:
        print("Error while grounding.")
        exit(-1)

    instance.line_pairings()

    for i, instrumented_ast in enumerate(instance.instrumented_asts):
        logger.debug('Instrumented program %d:\n%s', i, '\n'.join(str(x) for x in instrumented_ast))

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

    mcs_hit_counter = Counter()
    mcss_negative = OrderedSet()
    mcss_mfl = OrderedSet()
    mcss_positive = OrderedSet()

    for mfl in instance.all_mfl(instance.missing_models[0][0], i=0):
        print(mfl)

    for i in range(len(instance.asts)):
        # for model in instance.missing_models[i]:
        if instance.missing_models[i]:
            model = instance.missing_models[i][0]
            runhelper.timer_start('mcs.time')
            if not instance.config.skip_mcs_negative_non_relaxed:
                for mcs in instance.all_mcs(model, relaxed=False, i=i):
                    mcss_negative.append(mcs)
                    for rule in mcs:
                        mcs_hit_counter[rule] += 1
            if not instance.config.skip_mcs_negative_relaxed:
                for mcs in instance.all_mcs(model, relaxed=True, i=i):
                    mcss_negative.append(mcs)
                    for rule in mcs:
                        mcs_hit_counter[rule] += 1
            if instance.config.use_mfl:
                for mcs in instance.all_mfl(model, relaxed=False, i=i):
                    mcss_mfl.append(mcs)
                    for rule in mcs:
                        mcs_hit_counter[rule] += 1
            runhelper.timer_stop('mcs.time')
        if instance.extra_models[i]:
            runhelper.timer_start('mcs.time')
            if instance.config.use_mcs_positive:
                for mcs in instance.all_mcs(instance.extra_models[i], i=i, positive=True):
                    mcss_positive.append(mcs)
                    for rule in mcs:
                        mcs_hit_counter[rule] += 1
            runhelper.timer_stop('mcs.time')

    runhelper.log_any('mcss.negative.pre', [set(mcs) for mcs in mcss_negative])
    runhelper.log_any('mcss.mfl.pre', [set(mcs) for mcs in mcss_mfl])
    runhelper.log_any('mcss.positive.pre', [set(mcs) for mcs in mcss_positive])
    mcss = mcss_negative.union(mcss_mfl).union(mcss_positive)
    runhelper.log_any('mcss.all.pre', [set(mcs) for mcs in mcss])

    if not mcss:
        mcss = OrderedSet([frozenset()])  # search even if no MCS was found

    if not instance.config.skip_mcs_line_pairings:
        line_pairings = instance.line_pairings()
        if line_pairings:
            for a, b, cost in line_pairings:
                if cost <= 2 and cost != 0:
                    mcss = OrderedSet([mcs | {b} for mcs in mcss])

    if instance.config.use_sbfl:
        try:
            highlighter = AspHighlithingVisitor()

            highlighter.highlight(config.get().input_file, list(chain.from_iterable(instance.missing_models)))

            scorer = MaxScorer()
            rule_score_calculator = RuleScoreCalculator(highlighter.graph.weights, scorer)

            clingo.ast.parse_files([config.get().input_file], lambda x: rule_score_calculator.visit(x))

            selected_lines = set([rule for rule, score in sorted(rule_score_calculator.scorer.get_scores(), key=lambda x: x[1], reverse=True)[:1]])

            mcss = OrderedSet([mcs | selected_lines for mcs in mcss])
        except:
            pass

    runhelper.log_any('mcss', [set(mcs) for mcs in mcss])
    runhelper.log_any('mcss.hit.count', mcs_hit_counter)
    mcss_sorted = list(map(set, sorted(mcss, key=lambda mcs: sum(map(lambda rule: mcs_hit_counter[rule], mcs)), reverse=True)))
    runhelper.log_any('mcss.sorted', mcss_sorted)

    if instance.config.selfeval_lines is not None:
        mcs_union = set().union(*mcss)
        faulty_lines = set(instance.config.selfeval_lines)
        runhelper.log_any('mcss.union', mcs_union)
        runhelper.log_any('selfeval.lines', faulty_lines)
        if not faulty_lines and (not mcs_union or mcs_union == {()}):
            runhelper.log_any('fault.identified', 'Yes (no incorrect lines)')
        elif not faulty_lines:
            runhelper.log_any('fault.identified', 'No (no incorrect lines)')
        elif faulty_lines == mcs_union:
            runhelper.log_any('fault.identified', 'Yes')
        elif faulty_lines < mcs_union:
            if faulty_lines == set(mcss_sorted[0]):
                runhelper.log_any('fault.identified', 'Yes (first MCS)')
            elif any(map(lambda x: faulty_lines == set(x), mcss_sorted)):
                runhelper.log_any('fault.identified', 'Yes (not first MCS)')
            else:
                runhelper.log_any('fault.identified', 'Superset')
        elif faulty_lines.intersection(mcs_union):
            runhelper.log_any('fault.identified', 'Subset')
        else:
            if not mcs_union or mcs_union == {()}:
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

    if instance.config.simulate_fault_localization and instance.config.selfeval_lines is not None:
        mcss = OrderedSet([tuple(instance.config.selfeval_lines)])
        if not mcss:
            mcss = OrderedSet([()])  # search even if no MCS was found

    if instance.config.exit_after_fault_localization:
        for i in range(len(instance.asts)):
            instance.get_control(i=i)
        exit()

    predicates_before_skip = instance.constantCollector.predicates

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
            for mcs in sorted(mcss, key=lambda mcs: sum(map(lambda rule: mcs_hit_counter[rule], mcs)), reverse=True):
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
                            print('Solution found')
                            print(asp_prog)
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
