import math
import random
from typing import List, Generic, TypeVar

import runhelper
from formhe.asp.instance import Instance
from formhe.asp.synthesis.ASPSpecGenerator import ASPSpecGenerator
from formhe.asp.synthesis.AspInterpreter import AspInterpreter
from formhe.asp.synthesis.AspVisitor import AspVisitor

FaultType = TypeVar('FaultType')


class FaultLocalizer(Generic[FaultType]):

    def __init__(self, instance: Instance):
        self.instance = instance

    def fault_localize(self) -> List:
        raise NotImplementedError()

    def sort(self, fls: List[FaultType]) -> List[FaultType]:
        raise NotImplementedError()

    def report(self, sorted_fls: List[FaultType]):
        if self.instance.config.selfeval_lines is not None:
            mcs_union = set().union(*sorted_fls)
            faulty_lines = set(self.instance.config.selfeval_lines[0])  # todo
            runhelper.log_any('fl.union', mcs_union)
            runhelper.log_any('selfeval.lines', faulty_lines)
            if self.instance.config.selfeval_fix is not None:
                runhelper.log_any('selfeval.fix', self.instance.config.selfeval_fix)
            runhelper.log_any('selfeval.deleted.lines', self.instance.config.selfeval_deleted_lines)
            runhelper.log_any('selfeval.changes.generate', self.instance.config.selfeval_changes_generate)
            runhelper.log_any('selfeval.changes.generate.n', self.instance.config.selfeval_changes_generate_n)
            runhelper.log_any('selfeval.changes.generate.n.unique', self.instance.config.selfeval_changes_generate_n_unique)
            runhelper.log_any('selfeval.changes.test', self.instance.config.selfeval_changes_test)
            runhelper.log_any('selfeval.changes.test.n', self.instance.config.selfeval_changes_test_n)
            runhelper.log_any('selfeval.changes.test.n.unique', self.instance.config.selfeval_changes_test_n_unique)
            if not faulty_lines and (not mcs_union or mcs_union == {()}):
                runhelper.log_any('fault.identified', 'Yes (no incorrect lines)')

            elif not faulty_lines:
                runhelper.log_any('fault.identified', 'Wrong (no incorrect lines)')

            elif not mcs_union or mcs_union == {()}:
                runhelper.log_any('fault.identified', 'No (no lines identified)')

            elif set(sorted_fls[0]) == faulty_lines:
                runhelper.log_any('fault.identified', 'Yes (first MCS)')

            elif set(sorted_fls[0]) < faulty_lines:
                runhelper.log_any('fault.identified', 'Subset (first MCS)')

            elif set(sorted_fls[0]) > faulty_lines:
                runhelper.log_any('fault.identified', 'Superset (first MCS)')

            elif not set(sorted_fls[0]).isdisjoint(faulty_lines):
                runhelper.log_any('fault.identified', 'Not Disjoint (first MCS)')

            elif any(map(lambda x: set(x) == faulty_lines, sorted_fls)):
                runhelper.log_any('fault.identified', 'Yes (not first MCS)')

            elif any(map(lambda x: set(x) < faulty_lines, sorted_fls)):
                runhelper.log_any('fault.identified', 'Subset (not first MCS)')

            elif any(map(lambda x: set(x) > faulty_lines, sorted_fls)):
                runhelper.log_any('fault.identified', 'Superset (not first MCS)')

            elif any(map(lambda x: not set(x).isdisjoint(faulty_lines), sorted_fls)):
                runhelper.log_any('fault.identified', 'Not Disjoint (not first MCS)')

            else:
                runhelper.log_any('fault.identified', 'Wrong (wrong lines identified)')

            if self.instance.config.selfeval_lines is not None and self.instance.config.selfeval_fix is not None:
                partial_results = []
                for i, (lines, fix) in enumerate(zip(self.instance.config.selfeval_lines, self.instance.config.selfeval_fix)):
                    if len(lines) > 0 and fix is not None:
                        asp_interpreter = AspInterpreter(self.instance, self.instance.constantCollector.predicates.keys())
                        if asp_interpreter.test(fix):
                            partial_results.append('Yes')
                        else:
                            partial_results.append('No')
                if len(partial_results) > 0:
                    runhelper.log_any('fault.partial', partial_results)
