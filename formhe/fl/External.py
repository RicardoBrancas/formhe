from typing import List

from ordered_set import OrderedSet

from fl.FaultLocalizer import FaultLocalizer


class ExternalFL(FaultLocalizer):

    def fault_localize(self) -> List:
        mcss_sorted = OrderedSet([tuple(self.instance.config.external_fl)])
        if not mcss_sorted:
            mcss_sorted = OrderedSet([()])  # search even if no MCS was found

        return list(mcss_sorted)