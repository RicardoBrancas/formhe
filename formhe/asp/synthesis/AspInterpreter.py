from typing import Any

from formhe.asp.instance import Instance
from formhe.trinity.Visitor import PostOrderInterpreter
from formhe.utils import config, perf


class AspInterpreter(PostOrderInterpreter):

    def __init__(self, instance: Instance):
        self.instance = instance
        self.part_counter = 0

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            for pred_name in self.instance.constantCollector.predicates.keys():
                if name == f'eval_{pred_name}':
                    def tmp(node, args):
                        return self.eval_predicate(node, args, pred_name)

                    return tmp
            raise e

    def test(self, prog):
        current_part = self.part_counter
        self.part_counter += 1
        # statement = f'#external program_part_activator_{current_part}.\n:- ' + prog + f', program_part_activator_{current_part}.'
        # perf.timer_start(perf.GROUNDING_TIME)
        control = self.instance.get_control(':- ' + prog + '.')
        # control.add(f'program_part_{current_part}', [], statement)
        # control.ground([(f'program_part_{current_part}', [])])
        # control.assign_external(Function(f'program_part_activator_{current_part}', []), True)
        # perf.timer_stop(perf.GROUNDING_TIME)

        perf.timer_start(perf.TMP_TIME)
        for i, m in enumerate(self.instance.answer_sets_asm.get_bandits(config.get().n_gt_sols_checked)):
            res = control.solve(assumptions=m)
            if res.unsatisfiable:
                self.instance.answer_sets_asm.update_bandit(m, 1)
                perf.update_counter(perf.CANDIDATE_CHECKS_COUNTER, i + 1)
                perf.timer_stop(perf.TMP_TIME)
                return False
        perf.timer_stop(perf.TMP_TIME)
        perf.update_counter(perf.CANDIDATE_CHECKS_COUNTER, 'inf')

        perf.timer_start(perf.TMP_TIME2)
        with control.solve(yield_=True) as handle:
            i = 0
            for m in handle:
                i += 1
                m_ord = frozenset(m.symbols(atoms=True))
                if not self.instance.ground_truth.check_model(m_ord):
                    perf.counter_inc(perf.UNSAT)
                    perf.update_counter(perf.GT_CHECKS_COUNTER, i)
                    perf.timer_stop(perf.TMP_TIME2)
                    return False
                perf.counter_inc(perf.SAT)
                if i >= config.get().n_candidate_sols_checked:
                    break
        perf.timer_stop(perf.TMP_TIME2)
        perf.update_counter(perf.GT_CHECKS_COUNTER, 'inf')

        # perf.timer_start(perf.TMP_TIME)
        # for i, m in enumerate(self.instance.answer_sets_asm.get_bandits(config.get().n_gt_sols_checked - 1, 1)):
        #     res = control.solve(assumptions=m)
        #     if res.unsatisfiable:
        #         self.instance.answer_sets_asm.update_bandit(m, 1)
        #         perf.update_counter(perf.CANDIDATE_CHECKS_COUNTER, i + 1 + 1)
        #         perf.timer_stop(perf.TMP_TIME)
        #         return False
        # perf.timer_stop(perf.TMP_TIME)
        # perf.update_counter(perf.CANDIDATE_CHECKS_COUNTER, 'inf')

        return True

    def eval_predicate(self, node, args, name):
        return f'{name}({", ".join(args)})'

    def eval_and(self, node, args):
        return ', '.join(args)

    def eval_eq(self, node, args):
        return args[0] + ' == ' + args[1]

    def eval_neq(self, node, args):
        return args[0] + ' != ' + args[1]

    def eval_add(self, node, args):
        return args[0] + ' + ' + args[1]

    def eval_sub(self, node, args):
        return args[0] + ' - ' + args[1]

    def eval_mul(self, node, args):
        return args[0] + ' * ' + args[1]

    def eval_div(self, node, args):
        return args[0] + ' / ' + args[1]

    def eval_abs(self, node, args):
        return f'abs({args[0]})'

    def eval_not(self, node, args):
        return f'not {args[0]}'

    def eval_classical_not(self, node, args):
        return f'-{args[0]}'

    def eval_or(self, node, args):
        return ' || '.join(args)

    def eval_tuple(self, node, args):
        return '(' + ', '.join(args) + ')'

    def eval_PBool(self, value):
        if value:
            return '#true'
        else:
            return '#false'

    def eval_Int(self, value):
        return str(value)
