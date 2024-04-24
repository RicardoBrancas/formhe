from typing import List

import runhelper
from ordered_set import OrderedSet

from fl.FaultLocalizer import FaultLocalizer


class LineMatchingFL(FaultLocalizer):

    def fault_localize(self) -> List:
        runhelper.timer_start('fl.line.matching.time')
        line_pairings = self.instance.line_pairings()
        runhelper.timer_stop('fl.line.matching.time')

        if isinstance(line_pairings, list):
            runhelper.log_any('fl.pairing.line.scores', [(b, s) for a, b, s in line_pairings])

        mcs = OrderedSet()
        if line_pairings:
            for a, b, cost in line_pairings:
                if cost <= self.instance.config.line_matching_threshold and cost != 0:
                    mcs.add(b)

        return [tuple(mcs)]
