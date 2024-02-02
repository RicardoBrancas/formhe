import logging
from typing import List

from ordered_set import OrderedSet

from fl.External import ExternalFL
from fl.FaultLocalizer import FaultLocalizer
from fl.LineMatching import LineMatchingFL
from fl.MCS import MCSFL
from fl.SBFL import SBFL

logger = logging.getLogger('formhe.asp.fault_localization')


class CombinationFaultLocalizer(FaultLocalizer):

    def fault_localize(self) -> List:

        mcs_fl = MCSFL(self.instance)
        mcss = mcs_fl.fault_localize()

        if not mcss:
            mcss = OrderedSet([frozenset()])

        if not self.instance.config.skip_mcs_line_pairings:
            lm_fl = LineMatchingFL(self.instance)
            lm_result = lm_fl.fault_localize()
            for b in lm_result:
                mcss = OrderedSet([mcs | {b} for mcs in mcss])

        if self.instance.config.use_sbfl:
            sb_fl = SBFL(self.instance)
            sbfl_result = sb_fl.fault_localize()
            for b in sbfl_result:
                mcss = OrderedSet([mcs | {b} for mcs in mcss])

        mcss_sorted = self.sort(mcss)
        self.report(mcss_sorted)

        if self.instance.config.external_fl is not None:
            external_fl = ExternalFL(self.instance)
            logger.warning("External fault localization detected. Overwriting default fault localization method.")
            mcss_sorted = external_fl.fault_localize()
            self.report(mcss_sorted)

        return mcss_sorted
