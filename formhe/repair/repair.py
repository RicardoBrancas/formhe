import logging
from abc import ABC

import runhelper
from asp.instance import Instance
from formhe.exceptions.parser_exceptions import InstanceGroundingException

logger = logging.getLogger('formhe.repair')


# print('**Fix Suggestion**\n')
#
# if mcs:
#     print(f'You can try replacing the following line{"s" if len(statement_enumerator.current_preset_statements) > 1 else ""}:\n')
#     print('\n'.join(['\t' + str(line) for line in mcs]))
#     print('\nWith (the "?" are missing parts you should fill in):\n')
#     print('\n'.join(['\t' + str(line) for line in statement_enumerator.current_preset_statements]))
#     print()
# else:
#     print(f'You can try adding the following line{"s" if len(statement_enumerator.current_preset_statements) > 1 else ""} (the "?" are missing parts you should fill in):\n')
#     print('\n'.join(['\t' + str(line) for line in statement_enumerator.current_preset_statements]))
#     print()

class RepairModule(ABC):

    def __init__(self, instance: Instance):
        self.instance = instance
        self.interpreter = None

    def process_solution(self, asp_prog):
        logger.info('Solution found')
        runhelper.log_any('solution', asp_prog)
        print('Solution found')
        print(asp_prog)

    def test_candidate(self, asp_prog):
        if self.interpreter is None:
            logger.error('Interpreter is not inited')
            raise RuntimeError()

        try:
            evaluation_result = self.interpreter.test(asp_prog)
        except (RuntimeError, InstanceGroundingException):
            # print('!' + ''.join(map(lambda b: '1' if b else '0', enumerator.model_relaxation_values(enumerator.model))))
            runhelper.timer_stop('enum.fail.time')
            runhelper.tag_increment('eval.fail.programs')
            logger.warning('Failed to eval: %s', asp_prog)
            return False

        if evaluation_result:
            self.process_solution(asp_prog)
            return True

        return False

    def repair(self, fls, predicates):
        raise NotImplementedError
