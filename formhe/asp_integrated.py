#!/usr/bin/env python
import logging
from pathlib import Path

import runhelper
from fl.Combination import CombinationFaultLocalizer
from fl.Debug import DebugFL
from formhe.asp.instance import Instance
from formhe.exceptions.parser_exceptions import InstanceParseException, InstanceGroundingException
from formhe.utils import config
from repair.combined_repair import CombinedRepair

logger = logging.getLogger('formhe.asp.integrated')


def main(instance: Instance, iteration=0):
    runhelper.tag_increment("iteration")

    verification, verification_detail = instance.verify(do_print=iteration == 0)

    if iteration == 0:
        print(f'You have [+passed {sum(verification_detail)} test(s)+] and [-failed {len(verification_detail) - sum(verification_detail)} test(s)-].\n')

    if verification:
        print('No problems found\n')
        exit()

    if instance.config.selfeval_fix_test:
        instance.self_verify()

    if iteration == 0 and set(instance.predicates_used - instance.global_predicates_generated):
        print('The following predicates are used in a rule but are never generated: ' + ' '.join(map(str, set(instance.predicates_used - instance.global_predicates_generated))), "\n")

    runhelper.timer_start('fault.localization.time')

    if instance.config.simulate_fault_localization and instance.config.selfeval_lines is not None:
        fault_localizer = DebugFL(instance)
        flss = fault_localizer.fault_localize()
    else:
        fault_localizer = CombinationFaultLocalizer(instance)
        flss = fault_localizer.fault_localize()

    runhelper.timer_stop('fault.localization.time')

    if iteration == 0 and flss[0]:
        program_lines = instance.get_program_lines()
        print('\nSuggested lines where bug is likely located:  ')
        for line in flss[0]:
            if line is None or line >= len(program_lines):
                continue
            print(f'**[{line + 1}]**', '`' + program_lines[line] + '`', '  ')

    print("\n", flush=True)

    if instance.config.fl_watch_file:
        Path(instance.config.fl_watch_file).touch()

    if instance.config.exit_after_fault_localization:
        exit()

    res = False
    for fls in flss:
        repair_module = CombinedRepair(instance)
        res = repair_module.repair(fls, fault_localizer.missing_lines, iteration=iteration)
        if res:
            break
        logger.info("Repair for lines %s was unsuccessful", str(fls))

    return res


if __name__ == '__main__':
    logger.info('Starting FormHe ASP bug finder')
    logger.info('%s', config.get())
    runhelper.init_logger("formhe.runhelper")
    runhelper.register_sigterm_handler()

    logger.info('Loading instance from %s', config.get().input_file)

    try:
        instance = Instance(config.get().input_file, suppress_override_message=False)
    except InstanceParseException:
        print("Error while parsing input file.")
        exit(-1)
    except InstanceGroundingException:
        print("Error while grounding.")
        exit(-2)

    res = main(instance)

    if not res:
        print('Synthesis Failed')
        exit(-3)
    else:
        exit(0)
