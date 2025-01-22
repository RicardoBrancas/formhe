import logging
from itertools import repeat

from ordered_set import OrderedSet

import runhelper
from asp.instance import Instance
from asp.synthesis.spec_generator import ASPSpecGenerator
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
        self.current_depth = None
        self.current_n_mutations = None

    def get_program(self, trinity_program):
        try:
            asp_prog = self.interpreter.eval(trinity_program)
            return asp_prog
        except RuntimeError:
            runhelper.timer_stop('enum.fail.time')
            runhelper.tag_increment('eval.fail.programs')
            logger.warning('Failed to parse: %s', trinity_program)
            return None

    def process_solution(self, asp_prog, fls=None):
        super().process_solution(asp_prog, fls=fls)
        runhelper.log_any('solution.depth', self.current_depth)
        runhelper.log_any('solution.n_mutations', self.current_n_mutations)

    def repair(self, fls, missing_lines: bool = False, iteration=0) -> bool:
        for depth in range(config.get().minimum_depth, config.get().maximum_depth):
            self.current_depth = depth

            modified_instance = Instance(ast=[self.instance.raw_input], skips=fls, reference_instance=self.instance.reference, canon_instance=self.instance.canon, suppress_override_message=True)
            spec_generator = ASPSpecGenerator(modified_instance, modified_instance.config.extra_vars, self.instance.predicates.items())
            trinity_spec = spec_generator.trinity_spec
            asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)

            # bound_statements = []
            # for rule in modified_instance.constantCollector.not_skipped:
            #     node = asp_visitor.visit(rule)
            #     bound_statements.append(PresetStatement(node[0], node[1]))

            semi_bound_statements = []
            for rule in modified_instance.skipped_rules:
                node = asp_visitor.visit(rule)
                semi_bound_statements.append(PresetStatement(node.children[0], node.children[1].children))

            empty_statements = [PresetStatement(None, []) for _ in range(config.get().empty_statements)]

            self.interpreter = AspInterpreter(modified_instance)

            runhelper.timer_start('smt.enum.construction.time')
            enumerator = NextGenEnumerator(trinity_spec, depth,
                                           semi_bound_statements=semi_bound_statements + empty_statements,
                                           free_predicates=OrderedSet(self.instance.predicates.keys()) - modified_instance.predicates_generated,
                                           force_generate_predicates=modified_instance.predicates_used - modified_instance.predicates_generated,
                                           additional_body_roots=config.get().additional_body_nodes)

            enumerator.create_no_unsafe_vars_constraints(list(repeat(asp_visitor.free_vars, len(enumerator.statements))))

            runhelper.timer_stop('smt.enum.construction.time')

            logger.info('Unsupported predicates: %s', str(set(OrderedSet(self.instance.predicates.keys()) - modified_instance.predicates_generated)))
            logger.info('Needed predicates: %s', str(set(modified_instance.predicates_used - modified_instance.predicates_generated)))

            for n_mutations in range(config.get().minimum_mutations, config.get().maximum_mutations):
                self.current_n_mutations = n_mutations
                logger.info("Starting enum for MCS %s, with depth %d and %d mutations", str(fls), depth, n_mutations)

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
                        # print(asp_prog)
                        solved = self.test_candidate(asp_prog, fls)
                    else:
                        solved = False
                    enumerator.update()
                    runhelper.timer_stop('eval.time')

                    if solved:
                        return True

        return False
