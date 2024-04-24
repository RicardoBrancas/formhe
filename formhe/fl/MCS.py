from collections import Counter
from typing import List

import runhelper
from ordered_set import OrderedSet

from formhe.asp.instance import Instance

from fl.FaultLocalizer import FaultLocalizer


class MCSFL(FaultLocalizer):

    def __init__(self, instance: Instance):
        super().__init__(instance)
        self.mcs_hit_counter = Counter()
        self.mcss_negative_non_relaxed = OrderedSet()
        self.mcss_negative_relaxed = OrderedSet()
        self.mfl = OrderedSet()
        self.mcss_positive = OrderedSet()

    def fault_localize(self) -> List:
        runhelper.log_any('models.missing', [len(x) for x in self.instance.missing_models])
        runhelper.log_any('models.extra', [len(x) for x in self.instance.extra_models])

        for i in range(len(self.instance.asts)):
            for model in self.instance.missing_models[i]:
                # if instance.missing_models[i]:
                #     model = instance.missing_models[i][0]
                runhelper.timer_start('fl.mcs.time')
                if not self.instance.config.skip_mcs_negative_non_relaxed:
                    for mcs in self.instance.all_mcs(model, relaxed=False, i=i):
                        self.mcss_negative_non_relaxed.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mcs.time')
                runhelper.timer_start('fl.mcs.time')
                if not self.instance.config.skip_mcs_negative_relaxed:
                    for mcs in self.instance.all_mcs(model, relaxed=True, i=i):
                        self.mcss_negative_relaxed.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mcs.time')
                runhelper.timer_start('fl.mfl.time')
                if self.instance.config.use_mfl:
                    for mcs in self.instance.all_mfl(model, relaxed=False, i=i):
                        self.mfl.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mfl.time')

            if self.instance.extra_models[i]:
                runhelper.timer_start('fl.mcs.time')
                if self.instance.config.use_mcs_positive or ((not self.instance.config.skip_mcs_positive_conditional) and all(map(lambda x: len(x) == 0, self.instance.missing_models))):
                    for mcs in self.instance.all_mcs(self.instance.extra_models[i], i=i, positive=True):
                        self.mcss_positive.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mcs.time')

        runhelper.log_any('symbolic.atoms.first', len(self.instance.controls[0].symbolic_atoms))
        runhelper.log_any('symbolic.atoms.all', [len(ctl.symbolic_atoms) for ctl in self.instance.controls])
        runhelper.log_any('fl.mcss.hit.count', self.mcs_hit_counter)

        runhelper.log_any('fl.mcss.negative.non.relaxed', [set(mcs) for mcs in self.mcss_negative_non_relaxed])
        runhelper.log_any('fl.mcss.negative.relaxed', [set(mcs) for mcs in self.mcss_negative_relaxed])
        runhelper.log_any('fl.mcss.positive', [set(mcs) for mcs in self.mcss_positive])
        runhelper.log_any('fl.mfl', [set(mcs) for mcs in self.mfl])

        fls = self.mcss_negative_non_relaxed.union(self.mcss_negative_relaxed).union(self.mcss_positive).union(self.mfl)

        runhelper.log_any('fl.all', [set(fl) for fl in fls])
        
        return list(fls)
