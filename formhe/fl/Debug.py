from typing import List

from ordered_set import OrderedSet

from fl.FaultLocalizer import FaultLocalizer


class DebugFL(FaultLocalizer):

    def fault_localize(self) -> List:
        mcss_sorted = OrderedSet([tuple(self.instance.config.selfeval_lines[0])])
        if not mcss_sorted:
            mcss_sorted = OrderedSet([()])  # search even if no MCS was found

        return list(mcss_sorted)
