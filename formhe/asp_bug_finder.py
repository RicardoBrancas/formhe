#!/usr/bin/env python

import logging

from ordered_set import OrderedSet

from formhe.asp.instance import Instance
from formhe.utils import config, perf

logger = logging.getLogger('formhe.asp.bug_finder')


def main():
    logger.info('Starting FormHe ASP bug finder')
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

    if unsats_union:
        instance = Instance(config.get().input_file, skips=unsats_union)

        print('Your solution is not correct. There is likely one or more bugs in the following statements:\n')

        for elem in instance.constantCollector.skipped:
            print(elem)

    else:
        overconstrained = False
        underconstrained = False
        overconstrained_why = None
        underconstrained_why = None

        control = instance.get_control()
        with control.solve(yield_=True) as handle:
            for m in handle:
                m_ord = frozenset(m.symbols(shown=True))
                if not instance.ground_truth.check_model(m_ord):
                    underconstrained = True
                    underconstrained_why = m_ord
                    break

        control = instance.ground_truth.get_control()
        with control.solve(yield_=True) as handle:
            for m in handle:
                m_ord = frozenset(m.symbols(shown=True))
                if not instance.check_model(m_ord):
                    overconstrained = True
                    overconstrained_why = m_ord
                    break

        if underconstrained and overconstrained:
            print('Your solution is incorrect.')
        elif underconstrained:
            print('Your solution is likely underconstrained.')
        elif overconstrained:
            print('Your solution is likely overconstrained.')
        else:
            print('No problems found')

        if underconstrained_why:
            print('The following should not be a valid answer set, but it is in your solution:')
            for atom in underconstrained_why:
                print(atom, end='. ')
            print()

        if overconstrained_why:
            print('The following should be a valid answer set, but it is not in your solution:')
            for atom in overconstrained_why:
                print(atom, end='. ')
            print()


if __name__ == '__main__':
    main()
