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
        self.mcss_negative = OrderedSet()
        self.strong_mcss_negative = OrderedSet()
        self.mcss_mfl = OrderedSet()
        self.mcss_positive = OrderedSet()

    def fault_localize(self) -> List:
        runhelper.log_any('models.missing', [len(x) for x in self.instance.missing_models])
        runhelper.log_any('models.extra', [len(x) for x in self.instance.extra_models])

        for i in range(len(self.instance.asts)):
            for model in self.instance.missing_models[i]:
                # if instance.missing_models[i]:
                #     model = instance.missing_models[i][0]
                runhelper.timer_start('mcs.time')
                if not self.instance.config.skip_mcs_negative_non_relaxed:
                    for mcs in self.instance.all_mcs(model, relaxed=False, i=i):
                        self.mcss_negative.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('mcs.time')
                runhelper.timer_start('mcs.time')
                if not self.instance.config.skip_mcs_negative_relaxed:
                    for mcs in self.instance.all_mcs(model, relaxed=True, i=i):
                        self.mcss_negative.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('mcs.time')
                runhelper.timer_start('mcs.time')
                if self.instance.config.use_mfl:
                    for mcs in self.instance.all_mfl(model, relaxed=False, i=i):
                        self.mcss_mfl.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('mcs.time')
                runhelper.timer_start('mcs.time')
                if self.instance.config.use_mcs_strong_negative_non_relaxed:
                    for mcs in self.instance.all_weak_mcs(model, relaxed=False, i=i):
                        self.strong_mcss_negative.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('mcs.time')
                runhelper.timer_start('mcs.time')
                if self.instance.config.use_mcs_strong_negative_relaxed:
                    for mcs in self.instance.all_weak_mcs(model, relaxed=True, i=i):
                        self.strong_mcss_negative.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('mcs.time')

            if self.instance.extra_models[i]:
                runhelper.timer_start('mcs.time')
                if self.instance.config.use_mcs_positive:
                    for mcs in self.instance.all_mcs(self.instance.extra_models[i], i=i, positive=True):
                        self.mcss_positive.append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('mcs.time')

        runhelper.log_any('symbolic.atoms.first', len(self.instance.controls[0].symbolic_atoms))
        runhelper.log_any('symbolic.atoms.all', [len(ctl.symbolic_atoms) for ctl in self.instance.controls])

        runhelper.log_any('mcss.negative.pre', [set(mcs) for mcs in self.mcss_negative])
        runhelper.log_any('strong.mcss.negative.pre', [set(mcs) for mcs in self.strong_mcss_negative])
        runhelper.log_any('mcss.mfl.pre', [set(mcs) for mcs in self.mcss_mfl])
        runhelper.log_any('mcss.positive.pre', [set(mcs) for mcs in self.mcss_positive])
        mcss = self.mcss_negative.union(self.strong_mcss_negative).union(self.mcss_mfl).union(self.mcss_positive)
        runhelper.log_any('mcss.all.pre', [set(mcs) for mcs in mcss])
        
        return list(mcss)
