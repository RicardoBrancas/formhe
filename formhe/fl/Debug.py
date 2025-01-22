import logging
from typing import List

import runhelper
from ordered_set import OrderedSet

from fl.fault_localizer import FaultLocalizer

logger = logging.getLogger('formhe.asp.fault_localization')


class DebugFL(FaultLocalizer):

    def fault_localize(self) -> List:
        mcss_sorted = OrderedSet([tuple(self.instance.config.selfeval_lines[0])])
        if not mcss_sorted:
            mcss_sorted = OrderedSet([()])  # search even if no MCS was found

        runhelper.log_any("fl.simulated", mcss_sorted)

        return list(mcss_sorted)
