from collections import Counter, defaultdict
from typing import List

from ordered_set import OrderedSet

import runhelper
from fl.FaultLocalizer import FaultLocalizer
from formhe.asp.instance import Instance
from utils.dc import combine_improved


def union(*args):
    if len(args) > 0:
        base = args[0]
        for other in args[1:]:
            base = base.union(other)

        return base

    else:
        return []


class MCSFL(FaultLocalizer):

    def __init__(self, instance: Instance):
        super().__init__(instance)
        self.mcs_hit_counter = Counter()
        self.mcss_negative_non_relaxed = defaultdict(lambda: OrderedSet())
        self.mcss_negative_relaxed = defaultdict(lambda: OrderedSet())
        self.mfl = defaultdict(lambda: OrderedSet())
        self.mcss_positive = defaultdict(lambda: OrderedSet())

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
                        self.mcss_negative_non_relaxed[i].append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mcs.time')
                runhelper.timer_start('fl.mcs.time')
                if not self.instance.config.skip_mcs_negative_relaxed:
                    for mcs in self.instance.all_mcs(model, relaxed=True, i=i):
                        self.mcss_negative_relaxed[i].append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mcs.time')
                runhelper.timer_start('fl.mfl.time')
                if self.instance.config.use_mfl:
                    for mcs in self.instance.all_mfl(model, relaxed=False, i=i):
                        self.mfl[i].append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mfl.time')

            if self.instance.extra_models[i]:
                runhelper.timer_start('fl.mcs.time')
                if self.instance.config.use_mcs_positive or ((not self.instance.config.skip_mcs_positive_conditional) and all(map(lambda x: len(x) == 0, self.instance.missing_models))):
                    for mcs in self.instance.all_mcs(self.instance.extra_models[i], i=i, positive=True):
                        self.mcss_positive[i].append(mcs)
                        for rule in mcs:
                            self.mcs_hit_counter[rule] += 1
                runhelper.timer_stop('fl.mcs.time')

        runhelper.log_any('symbolic.atoms.first', len(self.instance.controls[0].symbolic_atoms))
        runhelper.log_any('symbolic.atoms.all', [len(ctl.symbolic_atoms) for ctl in self.instance.controls])
        runhelper.log_any('fl.mcss.hit.count', self.mcs_hit_counter)

        runhelper.log_any('fl.mcss.negative.non.relaxed', self.mcss_negative_non_relaxed)
        runhelper.log_any('fl.mcss.negative.relaxed', self.mcss_negative_relaxed)
        runhelper.log_any('fl.mcss.positive', self.mcss_positive)
        runhelper.log_any('fl.mfl', self.mfl)

        diags = []

        for i in range(len(self.instance.asts)):
            diags_for_i = OrderedSet()

            diags_for_i.update(self.mcss_negative_relaxed[i])
            diags_for_i.update(self.mcss_negative_non_relaxed[i])
            diags_for_i.update(self.mcss_positive[i])
            diags_for_i.update(self.mfl[i])

            tmp = OrderedSet([d for d in diags_for_i if len(d) > 0])
            if len(tmp) > 0:
                diags.append(tmp)

        runhelper.log_any("diags", diags)
        runhelper.timer_start("diags.combination")
        diags_combined = combine_improved(diags)
        runhelper.timer_stop("diags.combination")
        runhelper.log_any("diags.combined", diags_combined)
        runhelper.log_any("diags.different", set(diags_combined) != set([tuple(diag) for diag in union(*diags)]))

        # runhelper.log_any('fl.all', [set(fl) for fl in fls])

        return [frozenset(d) for d in diags_combined]
