from typing import List, Generic, TypeVar, NamedTuple

import runhelper
from formhe.asp.instance import Instance
from formhe.asp.synthesis.AspInterpreter import AspInterpreter

FaultType = TypeVar('FaultType')

SelfEvalResult = NamedTuple("SelfEvalResult", [("quality", int), ("selfeval_index", int), ("fault_identified", str)])


class FaultLocalizer(Generic[FaultType]):

    def __init__(self, instance: Instance):
        self.instance = instance
        self.missing_lines = None

    def fault_localize(self) -> List:
        raise NotImplementedError()

    def sort(self, fls: List[FaultType]) -> List[FaultType]:
        raise NotImplementedError()

    def report(self, sorted_fls: List[FaultType]):
        runhelper.log_any('fl.sorted', sorted_fls)
        if sorted_fls:
            runhelper.log_any('fl.first', sorted_fls[0])
        runhelper.log_any('fl.union', set().union(*sorted_fls))
        if self.instance.config.selfeval_lines is not None:
            runhelper.log_any('selfeval.lines', self.instance.config.selfeval_lines)
        if self.instance.config.selfeval_fix is not None:
            runhelper.log_any('selfeval.fix', self.instance.config.selfeval_fix)

        if self.instance.config.test_partial_fault:
            partial_results = []
            for i, (lines, fix) in enumerate(zip(self.instance.config.selfeval_lines, self.instance.config.selfeval_fix)):
                if len(lines) > 0 and fix is not None:
                    asp_interpreter = AspInterpreter(self.instance)
                    if asp_interpreter.test(fix):
                        partial_results.append('Yes')
                    else:
                        partial_results.append('No')
            if len(partial_results) > 0:
                runhelper.log_any('fault.partial', partial_results)

        if self.instance.config.selfeval_lines:
            selfeval_results: list[SelfEvalResult] = []
            for selfeval_i, selfeval_lines in enumerate(self.instance.config.selfeval_lines):
                selfeval_lines = set(selfeval_lines)
                if not selfeval_lines and not sorted_fls[0]:
                    selfeval_results.append(SelfEvalResult(11, selfeval_i, 'Yes (no incorrect lines)'))

                elif not selfeval_lines:
                    selfeval_results.append(SelfEvalResult(2, selfeval_i, 'Wrong (no incorrect lines)'))

                elif not sorted_fls[0]:
                    selfeval_results.append(SelfEvalResult(3, selfeval_i, 'No (no lines identified)'))

                elif set(sorted_fls[0]) == selfeval_lines:
                    selfeval_results.append(SelfEvalResult(12, selfeval_i, 'Yes (first MCS)'))

                elif set(sorted_fls[0]) < selfeval_lines:
                    selfeval_results.append(SelfEvalResult(9, selfeval_i, 'Subset (first MCS)'))

                elif set(sorted_fls[0]) > selfeval_lines:
                    selfeval_results.append(SelfEvalResult(10, selfeval_i, 'Superset (first MCS)'))

                elif not set(sorted_fls[0]).isdisjoint(selfeval_lines):
                    selfeval_results.append(SelfEvalResult(8, selfeval_i, 'Not Disjoint (first MCS)'))

                elif any(map(lambda x: set(x) == selfeval_lines, sorted_fls)):
                    selfeval_results.append(SelfEvalResult(7, selfeval_i, 'Yes (not first MCS)'))

                elif any(map(lambda x: set(x) < selfeval_lines, sorted_fls)):
                    selfeval_results.append(SelfEvalResult(5, selfeval_i, 'Subset (not first MCS)'))

                elif any(map(lambda x: set(x) > selfeval_lines, sorted_fls)):
                    selfeval_results.append(SelfEvalResult(6, selfeval_i, 'Superset (not first MCS)'))

                elif any(map(lambda x: not set(x).isdisjoint(selfeval_lines), sorted_fls)):
                    selfeval_results.append(SelfEvalResult(4, selfeval_i, 'Not Disjoint (not first MCS)'))

                else:
                    selfeval_results.append(SelfEvalResult(1, selfeval_i, 'Wrong (wrong lines identified)'))

            if selfeval_results:
                best_result = sorted(selfeval_results, key=lambda x: x.quality, reverse=True)[0]
                runhelper.log_any('selfeval.selected.index', best_result.selfeval_index)
                runhelper.log_any('fault.identified', best_result.fault_identified)
