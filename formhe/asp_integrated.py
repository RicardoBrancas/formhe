#!/usr/bin/env python

import logging
import traceback

from ordered_set import OrderedSet

from formhe.asp.synthesis.ASPSpecGenerator import ASPSpecGenerator
from formhe.asp.synthesis.AspInterpreter import AspInterpreter
from formhe.asp.synthesis.AspVisitor import AspVisitor
from formhe.asp.synthesis.StatementEnumerator import StatementEnumerator
from formhe.asp.instance import Instance
from formhe.trinity.z3_enumerator import Z3Enumerator
from formhe.utils import config
from formhe.utils import perf, algs

logger = logging.getLogger('formhe.asp.integrated')


def main():
    logger.info('Starting FormHe ASP integrated bug fixer')
    logger.info('%s', config.get())

    logger.info('Loading instance from %s', config.get().input_file)
    instance = Instance(config.get().input_file)

    logger.debug('Instrumented program:\n%s', '\n'.join(str(x) for x in instance.instrumented_ast))

    unsats_union = OrderedSet()

    perf.timer_start(perf.FAULT_LOCALIZATION_TIME)
    for unsat in instance.find_mcs():
        for var in unsat:
            unsats_union.add(var)
    perf.timer_stop(perf.FAULT_LOCALIZATION_TIME)

    instance = Instance(config.get().input_file, skips=unsats_union)

    for a in instance.constantCollector.skipped:
        print(a)

    spec_generator = ASPSpecGenerator(instance, 2)
    trinity_spec = spec_generator.trinity_spec

    asp_visitor = AspVisitor(trinity_spec)

    if len(instance.constantCollector.skipped) > 1:
        raise NotImplementedError()

    preset_atoms = []
    for rule in instance.constantCollector.skipped:
        preset_atoms.append(asp_visitor.visit(rule))
        print(preset_atoms[-1])

    asp_interpreter = AspInterpreter(instance)
    perf.timer_start(perf.ANSWER_SET_ENUM_TIME)
    if not config.get().no_semantic_constraints:
        instance.find_wrong_models(max_sols=1000)
    instance.generate_correct_models(config.get().n_gt_sols_generated)
    perf.timer_stop(perf.ANSWER_SET_ENUM_TIME)

    sorted_cores = sorted(instance.cores, key=len)

    hammings = []
    for m_a in instance.answer_sets:
        str_a = ' '.join([str(x) for x in m_a])
        for m_b in instance.answer_sets:
            if m_b != m_a:
                str_b = ' '.join([str(x) for x in m_b])
                hammings.append(algs.hamming(str_a, str_b))

    if len(hammings) > 0:
        logger.info('Average hamming distance of ground truth models: %f', sum(hammings) / len(hammings))

    solved = False
    enums = 0
    for depth in range(config.get().minimum_depth, config.get().maximum_depth):
        # perf.timer_start(perf.TMP_TIME)
        atom_enum_constructor = lambda n, p: Z3Enumerator(trinity_spec, depth, n, predicates_names=instance.constantCollector.predicates.keys(), cores=sorted_cores, free_vars=asp_visitor.free_vars,
                                                          preset_atoms=p)
        statement_enumerator = StatementEnumerator(atom_enum_constructor, preset_atoms[0], 1, asp_visitor.free_vars, depth)
        # perf.timer_stop(perf.TMP_TIME)

        perf.timer_start(perf.EVAL_FAIL_TIME)
        while prog := next(statement_enumerator):
            enums += 1
            # if enums >= 5000:
            #     perf.log()
            #     print(instance.answer_sets_asm)
            #     enums = 0
            perf.counter_inc(perf.ENUM_PROGRAMS)
            perf.timer_start(perf.EVAL_TIME)
            try:
                asp_prog = asp_interpreter.eval(prog)
                # logger.debug(asp_prog)
                # print(asp_prog)

                res = asp_interpreter.test(asp_prog)
                if res:
                    logger.info('Solution found')
                    print(asp_prog)
                    solved = True
                    break
            except RuntimeError as e:
                perf.timer_stop(perf.EVAL_FAIL_TIME)
                perf.counter_inc(perf.EVAL_FAIL_PROGRAMS)
                logger.warning('Failed to parse: %s', prog)
                traceback.print_exception(e)
                # print('EVAL EXCEPTION')
            except Exception as e:
                # traceback.print_exception(e)
                raise e
            perf.timer_stop(perf.EVAL_TIME)
            perf.timer_start(perf.EVAL_FAIL_TIME)

        if solved:
            break

    logger.info(instance.ground_truth.check_model.cache_info())

    # print()
    # print('Modified program:')
    # print()
    #
    # for node in instance.ast:
    #     print(node)
    #
    # print()
    #
    # instance.find_wrong_models(max_sols=args.model_attempts)
    # instance.generate_correct_models(max_sols=args.gt_model_attempts)
    #
    # sygus = SyGuSVisitor(instance, instance.cores, instance.answer_sets,
    #                      relax_pbe_constraints=args.relax_pbe_constraints, constrain_reflexive=args.constrain_reflexive,
    #                      skip_cores=args.skip_cores)
    # sygus.solve(max_cores=args.max_cores, max_models=args.max_models, skip_cores=args.skip_cores)


if __name__ == '__main__':
    main()
