import logging
import random
from typing import List, Iterable
import math

import bentoml
import httpx
import runhelper
from ordered_set import OrderedSet

from asp.instance import Instance
from fl.External import ExternalFL
from fl.FaultLocalizer import FaultLocalizer, FaultType
from fl.LLM import LLMFaultLocalizer
from fl.LineMatching import LineMatchingFL
from fl.MCS import MCSFL
from fl.SBFL import SBFL

logger = logging.getLogger('formhe.asp.fault_localization')


def avg(collection: Iterable):
    l = list(collection)
    return sum(l) / len(l)


class CombinationFaultLocalizer(FaultLocalizer):

    def __init__(self, instance: Instance):
        super().__init__(instance)
        self.mcs_hit_counter = None
        self.llm_line_scores = None

    def sort(self, fls: List[FaultType]) -> List[FaultType]:
        runhelper.log_any('fl.unsorted', [set(fl) for fl in fls])
        if self.llm_line_scores is not None:
            mcs_hit_counter = self.llm_line_scores
        elif self.mcs_hit_counter is not None:
            mcs_hit_counter = self.mcs_hit_counter
        else:
            logger.error("Tried to sort MCSs with no sort information available")
        match self.instance.config.mcs_sorting_method:
            case 'none':
                mcss_sorted = list(map(set, fls))
            case 'none-smallest':
                mcss_sorted = list(map(set, sorted(fls, key=lambda mcs: len(mcs) if len(mcs) != 0 else math.inf)))
            case 'hit-count':
                mcss_sorted = list(map(set, sorted(fls, key=lambda mcs: sum(map(lambda rule: mcs_hit_counter[rule], mcs)), reverse=True)))
            case 'hit-count-avg':
                mcss_sorted = list(map(set, sorted(fls, key=lambda mcs: avg(map(lambda rule: mcs_hit_counter[rule], mcs)), reverse=True)))
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

        runhelper.log_any('fl.sorted', mcss_sorted)
        return mcss_sorted

    def fault_localize(self) -> List:

        mcs_fl = MCSFL(self.instance)
        mcss = mcs_fl.fault_localize()
        self.mcs_hit_counter = mcs_fl.mcs_hit_counter

        if not self.instance.config.skip_llm_fl:
            try:
                with bentoml.SyncHTTPClient(self.instance.config.llm_url, timeout=600) as client:
                    llm_fl = LLMFaultLocalizer(self.instance, client)
                    llm_result = llm_fl.fault_localize()
                    self.llm_line_scores = llm_fl.scores
                    mcss += llm_result
            except (httpx.ConnectError, httpx.ReadTimeout):
                logger.warning("Could not connect to LLM server...")

        if not mcss:
            mcss = [frozenset()]

        if not self.instance.config.skip_mcs_line_pairings:
            lm_fl = LineMatchingFL(self.instance)
            lm_result = lm_fl.fault_localize()
            assert len(lm_result) == 1
            for b in lm_result[0]:
                mcss = OrderedSet([mcs | {b} for mcs in mcss])

        if self.instance.config.use_sbfl:
            sb_fl = SBFL(self.instance)
            sbfl_result = sb_fl.fault_localize()
            assert len(sbfl_result) == 1
            for b in sbfl_result[0]:
                mcss = OrderedSet([mcs | {b} for mcs in mcss])

        mcss_sorted = self.sort(mcss)
        self.report(mcss_sorted)

        if self.instance.config.external_fl is not None:
            external_fl = ExternalFL(self.instance)
            logger.warning("External fault localization detected. Overwriting default fault localization method.")
            mcss_sorted = external_fl.fault_localize()
            self.report(mcss_sorted)

        return mcss_sorted
