import logging
import re
from typing import Any

import runhelper

from formhe.asp.instance import Instance
from formhe.trinity.Visitor import PostOrderInterpreter
from formhe.utils import config, perf

logger = logging.getLogger('formhe.asp.interpreter')


class AspInterpreter(PostOrderInterpreter):

    def __init__(self, instance: Instance, predicates: list = None):
        self.instance = instance
        self.part_counter = 0
        if predicates is None:
            self.predicates = self.instance.constantCollector.predicates.keys()
        else:
            self.predicates = predicates

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            pred_name = name.removeprefix('eval_')
            if match := re.fullmatch(r'([a-zA-Z_0-9]+)/\d+', pred_name):
                def tmp(node, args):
                    return self.eval_predicate(node, args, match.group(1))

                return tmp

            else:
                raise NotImplementedError(f'Unknown function {pred_name}!')

    def test(self, prog):
        for control_i in range(len(self.instance.asts)):
            control = self.instance.get_control(prog, project=True, clingo_args=['--opt-mode=optN'] if self.instance.config.optimization_problem else [], i=control_i)
            runhelper.timer_start('interpreter.test.time')
            at_least_one = False
            with control.solve(yield_=True) as handle:
                for m in handle:
                    if self.instance.config.optimization_problem and not m.optimality_proven:
                        continue

                    model = tuple(sorted((m.symbols(shown=True))))
                    if model not in self.instance.ground_truth.models[control_i]:
                        runhelper.tag_increment('interpreter.test.early.exit')
                        runhelper.timer_stop('interpreter.test.time')
                        return False

                    at_least_one = True

            if not at_least_one:
                runhelper.tag_increment('interpreter.test.empty.exit')
                runhelper.timer_stop('interpreter.test.time')
                return False

            runhelper.timer_stop('interpreter.test.time')
        return True

    def eval_predicate(self, node, args, name):
        return f'{name}({", ".join(args)})'

    def eval_and(self, node, args):
        return ', '.join([arg for arg in args if arg.strip() != ""])

    def eval_and_(self, node, args):
        return ', '.join([arg for arg in args if arg.strip() != ""])

    def eval_pool(self, node, args):
        return '; '.join([arg for arg in args if arg.strip() != ""])

    def eval_stmt_and(self, node, args):
        return ' '.join([arg for arg in args if arg.strip() != ""])

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
        return ' | '.join(args)

    def eval_tuple(self, node, args):
        return '(' + ', '.join(args) + ')'

    def eval_stmt(self, node, args):
        if not args[0] and not args[1]:
            return ""
        if args[1]:
            return args[0] + ' :- ' + args[1] + '.'
        else:
            return args[0] + '.'

    def eval_minimize(self, node, args):
        return ':~ ' + args[3] + '. [' + args[0] + '@' + args[2] + ']'

    def eval_aggregate(self, node, args):
        return args[0] + ' { ' + args[1] + ' : ' + args[2] + ' } ' + args[3]

    def eval_aggregate_pool(self, node, args):
        return args[0] + ' { ' + args[1] + ' } ' + args[2]

    def eval_interval(self, node, args):
        return f'({args[0]}..{args[1]})'

    def eval_empty(self, node, args):
        return ''

    def eval_define(self, node, args):
        return f'#const {args[0]} = {args[1]}.'

    def eval_PBool(self, value):
        if value:
            return '#true'
        else:
            return '#false'

    def eval_Terminal(self, value):
        return str(value)

    def eval_BodyAggregateFunc(self, value):
        return value
