import logging
from abc import ABC
from difflib import SequenceMatcher
from itertools import zip_longest, product

import numpy as np

import formhe.utils.clingo
import runhelper
from asp.synthesis.AspInterpreter import AspInterpreter
from formhe.asp.instance import Instance
from formhe.asp.synthesis.AspVisitor import AspVisitor
from formhe.asp.synthesis.spec_generator import ASPSpecGenerator
from formhe.exceptions.parser_exceptions import InstanceGroundingException
from formhe.trinity.DSL import Node, ApplyNode, AtomNode
from trinity.DSL.node import HoleNode

logger = logging.getLogger('formhe.repair')


# print('**Fix Suggestion**\n')
#
# if mcs:
#     print(f'You can try replacing the following line{"s" if len(statement_enumerator.current_preset_statements) > 1 else ""}:\n')
#     print('\n'.join(['\t' + str(line) for line in mcs]))
#     print('\nWith (the "?" are missing parts you should fill in):\n')
#     print('\n'.join(['\t' + str(line) for line in statement_enumerator.current_preset_statements]))
#     print()
# else:
#     print(f'You can try adding the following line{"s" if len(statement_enumerator.current_preset_statements) > 1 else ""} (the "?" are missing parts you should fill in):\n')
#     print('\n'.join(['\t' + str(line) for line in statement_enumerator.current_preset_statements]))
#     print()

def node_is_non_empty(node):
    return not (node is None or (isinstance(node, ApplyNode) and node.name == "empty"))


def tree_diff(original_node: Node, correction_node: Node):
    if isinstance(original_node, ApplyNode) and isinstance(correction_node, ApplyNode):
        if original_node.name == correction_node.name:
            return ApplyNode(original_node.production, [tree_diff(a, b) for a, b in zip_longest(original_node.children, correction_node.children, fillvalue=None)])

    elif isinstance(original_node, AtomNode) and isinstance(correction_node, AtomNode):
        if original_node.production == correction_node.production:
            return original_node

    if original_node is None and isinstance(correction_node, ApplyNode) and correction_node.name == "stmt":
        return ApplyNode(correction_node.production, [HoleNode() if node_is_non_empty(correction_node.children[0]) else correction_node.children[0],
                                                      HoleNode() if node_is_non_empty(correction_node.children[1]) else correction_node.children[1]])

    if correction_node is None and isinstance(original_node, ApplyNode) and original_node.name == "stmt":
        return ApplyNode(original_node.production, [HoleNode() if node_is_non_empty(original_node.children[0]) else original_node.children[0],
                                                    HoleNode() if node_is_non_empty(original_node.children[1]) else original_node.children[1]])

    return HoleNode()


class RepairModule(ABC):

    def __init__(self, instance: Instance):
        self.instance = instance
        self.interpreter = None

    def process_solution(self, asp_prog, fls=None):
        if not self.instance.config.hints_only:
            logger.info('Solution found')
            runhelper.log_any('solution', asp_prog)
            runhelper.log_any('solution.by', self.__class__.__name__)
            print('Solution found')
            print(asp_prog)

        else:
            print('**Fix Suggestion**\n')
            if fls:
                print(f'You can try replacing the line{"s" if len(fls) > 1 else ""} identified above with (the "?" are holes you should fill in or remove):\n')
                self.print_hint(original=fls, correction=asp_prog)
            else:
                print(f'You can try adding the following line{"s" if len(asp_prog.splitlines()) > 1 else ""} (the "?" are holes you should fill in or remove):\n')
                self.print_hint(original=fls, correction=asp_prog)

    def print_hint(self, original, correction):
        from scipy.optimize import linear_sum_assignment

        logger.debug("To be replaced: %s", original)
        logger.debug("Replacement: %s", correction)

        spec_generator = ASPSpecGenerator(self.instance, 0, self.instance.predicates.items())
        asp_visitor = AspVisitor(spec_generator.trinity_spec, spec_generator.free_vars, domain_predicates=self.instance.problem.output_predicates_tuple + self.instance.problem.input_predicates_tuple)
        asp_interpreter = AspInterpreter(self.instance)

        correction_nodes = [asp_visitor.visit(formhe.utils.clingo.parse_string(l)[0]) for l in correction.splitlines()]
        original_nodes = [asp_visitor.visit(l) for l in original]

        if len(correction_nodes) < len(original_nodes):
            correction_nodes += [None] * (len(original_nodes) - len(correction_nodes))

        if len(original_nodes) < len(correction_nodes):
            original_nodes += [None] * (len(correction_nodes) - len(original_nodes))

        correction_strs = [asp_interpreter.eval(n) if n else "" for n in correction_nodes]
        original_strs = [asp_interpreter.eval(n) if n else "" for n in original_nodes]

        costs = np.zeros((len(original_strs), len(correction_strs)))
        for (i, a), (j, b) in product(enumerate(original_strs), enumerate(correction_strs)):
            costs[i, j] = SequenceMatcher(None, a, b).ratio()

        row_ind, col_ind = linear_sum_assignment(costs, maximize=True)

        for a, b in zip(row_ind, col_ind):
            logger.debug(original_strs[a] + ' BECOMES ' + correction_strs[b])
            print('\t' + asp_interpreter.eval(tree_diff(original_nodes[a], correction_nodes[b])))

    def test_candidate(self, asp_prog, fls=None):
        if self.interpreter is None:
            logger.error('Interpreter is not inited')
            raise RuntimeError()

        try:
            evaluation_result = self.interpreter.test(asp_prog)
        except (RuntimeError, InstanceGroundingException):
            # print('!' + ''.join(map(lambda b: '1' if b else '0', enumerator.model_relaxation_values(enumerator.model))))
            runhelper.timer_stop('enum.fail.time')
            runhelper.tag_increment('eval.fail.programs')
            logger.warning('Failed to eval: %s', asp_prog)
            return False

        if evaluation_result:
            self.process_solution(asp_prog, fls=[l for i, l in enumerate(self.instance.base_ast) if i in fls])
            return True

        return False

    def repair(self, fls, predicates):
        raise NotImplementedError
