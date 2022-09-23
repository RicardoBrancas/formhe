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

        print('You solution is not correct. There is likely one or more bugs in the following statements:\n')

        for elem in instance.constantCollector.skipped:
            print(elem)

    else:
        print('No problems found')


if __name__ == '__main__':
    main()
