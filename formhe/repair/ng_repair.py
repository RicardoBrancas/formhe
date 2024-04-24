import logging
from itertools import repeat

from ordered_set import OrderedSet

import runhelper
from asp.instance import Instance
from asp.synthesis.ASPSpecGenerator import ASPSpecGenerator
from asp.synthesis.AspInterpreter import AspInterpreter
from asp.synthesis.AspVisitor import AspVisitor
from repair.repair import RepairModule
from trinity.ng_enumerator import PresetStatement, NextGenEnumerator
from utils import config
from utils.perm import PermutationGeneratorHelper

logger = logging.getLogger('formhe.repair')


class NextGenRepair(RepairModule):

    def __init__(self, instance: Instance):
        super().__init__(instance)

    def get_program(self, trinity_program):
        try:
            asp_prog = self.interpreter.eval(trinity_program)
            return asp_prog
        except RuntimeError:
            runhelper.timer_stop('enum.fail.time')
            runhelper.tag_increment('eval.fail.programs')
            logger.warning('Failed to parse: %s', trinity_program)
            return None

    def repair(self, fls, predicates) -> bool:
        for depth in range(config.get().minimum_depth, config.get().maximum_depth):
            for mcs in fls:

                modified_instance = Instance(config.get().input_file, skips=mcs, ground_truth_instance=self.instance.ground_truth, suppress_override_message=True)
                spec_generator = ASPSpecGenerator(modified_instance, modified_instance.config.extra_vars, predicates.items())
                trinity_spec = spec_generator.trinity_spec
                asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)

                # bound_statements = []
                # for rule in modified_instance.constantCollector.not_skipped:
                #     node = asp_visitor.visit(rule)
                #     bound_statements.append(PresetStatement(node[0], node[1]))

                semi_bound_statements = []
                for rule in modified_instance.constantCollector.skipped:
                    node = asp_visitor.visit(rule)
                    semi_bound_statements.append(PresetStatement(node.children[0], node.children[1].children))

                empty_statements = [PresetStatement(None, []),
                                    PresetStatement(None, [])]

                self.interpreter = AspInterpreter(modified_instance, predicates.keys())

                runhelper.timer_start('smt.enum.construction.time')
                enumerator = NextGenEnumerator(trinity_spec, depth,
                                               semi_bound_statements=semi_bound_statements + empty_statements, free_predicates=OrderedSet(predicates.keys()) - modified_instance.constantCollector.predicates_generated,
                                               force_generate_predicates=modified_instance.constantCollector.predicates_used - modified_instance.constantCollector.predicates_generated,
                                               additional_body_roots=1)

                enumerator.create_no_unsafe_vars_constraints(list(repeat(asp_visitor.free_vars, len(enumerator.statements))))

                runhelper.timer_stop('smt.enum.construction.time')

                logger.info('Unsupported predicates: %s', str(set(OrderedSet(predicates.keys()) - modified_instance.constantCollector.predicates_generated)))
                logger.info('Needed predicates: %s', str(set(modified_instance.constantCollector.predicates_used - modified_instance.constantCollector.predicates_generated)))

                for n_mutations in range(config.get().minimum_mutations, config.get().maximum_mutations):
                    logger.info("Starting enum for MCS %s, with depth %d and %d mutations", str(mcs), depth, n_mutations)

                    perm_helper = PermutationGeneratorHelper(config.get().seed, len(enumerator.relaxation_vars()), n_mutations, enumerator)

                    while True:
                        runhelper.timer_start('enum.z3.time')
                        prog = perm_helper.next()
                        runhelper.timer_stop('enum.z3.time')

                        if prog is None:
                            break

                        runhelper.tag_increment('enum.programs')
                        runhelper.timer_start('eval.time')
                        asp_prog = self.get_program(prog)
                        if asp_prog is not None:
                            solved = self.test_candidate(asp_prog)
                        else:
                            solved = False
                        enumerator.update()
                        runhelper.timer_stop('eval.time')

                        if solved:
                            runhelper.log_any('solution.n_mutations', n_mutations)
                            return True

        return False
