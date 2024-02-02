from typing import List

from ordered_set import OrderedSet

from fl.FaultLocalizer import FaultLocalizer


class LineMatchingFL(FaultLocalizer):

    def fault_localize(self) -> List:
        line_pairings = self.instance.line_pairings()
        mcs = OrderedSet()
        if line_pairings:
            for a, b, cost in line_pairings:
                if cost <= 2 and cost != 0:
                    mcs.add(b)

        return [tuple(mcs)]
