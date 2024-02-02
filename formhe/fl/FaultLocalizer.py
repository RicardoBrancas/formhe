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
        runhelper.log_any('mcss', [set(fl) for fl in fls])
        if hasattr(self, 'mcs_hit_counter'):
            mcs_hit_counter = self.mcs_hit_counter
        else:
            mcs_hit_counter = None  # todo there is a better way to do this for sure
        runhelper.log_any('mcss.hit.count', mcs_hit_counter)
        match self.instance.config.mcs_sorting_method:
            case 'none':
                mcss_sorted = list(map(set, fls))
            case 'none-smallest':
                mcss_sorted = list(map(set, sorted(fls, key=lambda mcs: len(mcs) if len(mcs) != 0 else math.inf)))
            case 'hit-count':
                mcss_sorted = list(map(set, sorted(fls, key=lambda mcs: sum(map(lambda rule: mcs_hit_counter[rule], mcs)), reverse=True)))
            case 'hit-count-normalized':
                mcss_sorted = list(map(set, sorted(fls, key=lambda mcs: sum(map(lambda rule: mcs_hit_counter[rule], mcs)) / len(mcs) if len(mcs) != 0 else math.inf, reverse=True)))
            case 'hit-count-smallest':
                mcss_sorted = list(map(set, sorted(fls, key=lambda mcs: (-len(mcs) if len(mcs) != 0 else -math.inf, sum(map(lambda rule: mcs_hit_counter[rule], mcs))), reverse=True)))
            case 'random':
                random_instance = random.Random(self.instance.config.seed)
                mcss_sorted = list(map(set, sorted(fls, key=lambda mcs: random_instance.random())))
            case 'random-smallest':
                random_instance = random.Random(self.instance.config.seed)
                mcss_sorted = list(map(set, sorted(fls, key=lambda mcs: (len(mcs) if len(mcs) != 0 else math.inf, random_instance.random()))))
            case _:
                raise NotImplementedError(f'Unrecognized MCS sorting method {self.instance.config.mcs_sorting_method}')

        return mcss_sorted

    def report(self, sorted_fls: List[FaultType]):
        if self.instance.config.selfeval_lines is not None:
            mcs_union = set().union(*sorted_fls)
            faulty_lines = set(self.instance.config.selfeval_lines)
            runhelper.log_any('mcss.sorted', sorted_fls)
            runhelper.log_any('mcss.union', mcs_union)
            runhelper.log_any('selfeval.lines', faulty_lines)
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

            if len(self.instance.config.selfeval_lines) > 0 and self.instance.config.selfeval_fix_test and self.instance.config.selfeval_fix is not None:
                spec_generator = ASPSpecGenerator(self.instance.ground_truth, 0, self.instance.constantCollector.predicates.items())
                trinity_spec = spec_generator.trinity_spec
                asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)
                asp_interpreter = AspInterpreter(self.instance, self.instance.constantCollector.predicates.keys())

                if asp_interpreter.test(self.instance.config.selfeval_fix):
                    runhelper.log_any('fault.partial', 'Yes')
                else:
                    runhelper.log_any('fault.partial', 'No')
