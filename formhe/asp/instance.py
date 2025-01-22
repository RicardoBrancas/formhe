import copy
import logging
import random
import re
import typing
from functools import cached_property
from pathlib import Path

import clingo.ast
from clingo import Control
from ordered_set import OrderedSet

import formhe.utils.clingo
import runhelper
from formhe.asp.problem import Problem
from formhe.asp.utils import Visitor, Instrumenter
from formhe.exceptions.parser_exceptions import InstanceGroundingException, InstanceParseException
from formhe.utils import config, iterutils

logger = logging.getLogger('formhe.asp.instance')


class Instance:

    def __init__(self,
                 filename: str = None, ast: clingo.ast.AST = None, skips: list[int] = None, parent_config: config.Config = None, reference_instance=None, suppress_override_message=True, is_groundtruth=False,
                 canon_instance=None, shuffle: typing.Union[bool, list[int]] = False):

        if (filename is not None and ast is not None) or (filename is None and ast is None):
            raise ValueError("Need to supply exactly one of {filename, ast}.")

        if filename is not None:
            self.filename = filename
            with open(filename) as f:
                self.raw_input = f.read()  # + '\n'.join(config.stdin_content)
        elif ast is not None:
            self.raw_input = '\n'.join(map(str, ast))

        self.clean_input = []
        for line in self.raw_input.splitlines():
            if line.startswith("%formhe-") or line.strip() == "":
                continue
            self.clean_input.append(line)
        if isinstance(shuffle, bool) and shuffle:
            self.shuffling = list(range(len(self.clean_input)))
            random.shuffle(self.shuffling)
            self.clean_input = [self.clean_input[i] for i in self.shuffling]
        elif isinstance(shuffle, list):
            self.clean_input = [self.clean_input[i] for i in shuffle]
        self.clean_input_string = "\n".join(self.clean_input)

        if not parent_config:
            self.config = copy.copy(config.get())
        else:
            self.config = copy.copy(parent_config)

        matches = re.findall(r'%formhe-([a-zA-Z0-9-_]*?):(.*)', self.raw_input)
        compact_matches = dict()
        for key, value in matches:
            key = key.replace('-', '_')
            if key in compact_matches:
                compact_matches[key] += ' ' + value
            else:
                compact_matches[key] = value
        self.config.process_overrides(compact_matches, suppress_override_message)

        if self.config.problem is None:
            raise ValueError("Problem type is undefined. Please use --problem or define in the instance file using %formhe-problem")

        self.problem = Problem.from_yaml_file(str((Path(self.config.problems_folder) / (self.config.problem + ".yaml")).absolute()))

        self.full_programs = []
        if self.problem.inputs:
            for input in self.problem.inputs:
                self.full_programs.append(self.clean_input_string + '\nformhe_definition_begin.\n' + input + '\nformhe_definition_end.')
        else:
            self.full_programs.append(self.clean_input_string)

        if self.problem.output_predicates:
            for pred in self.problem.output_predicates:
                for i, input in enumerate(self.full_programs):
                    self.full_programs[i] = input + f'\n#show {pred}.'

        global_constant_collector = Visitor()
        self.asts = []
        self.instrumented_asts = []
        try:
            ast = formhe.utils.clingo.parse_string(self.clean_input_string)
            base_constant_collector = Visitor(skips)
            self.instrumenter = Instrumenter()
            [global_constant_collector.visit(copy.copy(node)) for node in ast]
            self.base_ast = iterutils.drop_nones([base_constant_collector.visit(copy.copy(node)) for node in ast])
            [self.instrumenter.visit(copy.copy(node)) for node in ast]
            for prog in self.full_programs:
                ast = formhe.utils.clingo.parse_string(prog)
                constant_collector = Visitor(skips)
                instrumenter = Instrumenter()
                [global_constant_collector.visit(copy.copy(node)) for node in ast]
                self.asts.append(iterutils.drop_nones([constant_collector.visit(copy.copy(node)) for node in ast]))
                self.instrumented_asts.append(iterutils.drop_nones([instrumenter.visit(node) for node in ast]))
        except Exception as e:
            raise InstanceParseException(e)

        self.constants = OrderedSet(self.problem.input_predicates_zero_arity) | global_constant_collector.constants
        self.definitions = OrderedSet(self.problem.input_constants) | global_constant_collector.definitions
        self.predicates = global_constant_collector.predicates
        self.predicates_used = base_constant_collector.predicates_used
        self.predicates_generated = base_constant_collector.predicates_generated
        self.global_predicates_generated = global_constant_collector.predicates_generated
        self.skipped_rules = base_constant_collector.skipped
        self.models: list[typing.Optional[OrderedSet]] = [None] * len(self.asts)
        self.controls = [self.get_control(i=i) for i in range(len(self.asts))]

        if canon_instance is not None:
            self.canon = canon_instance
        elif not is_groundtruth:
            config_tmp = copy.copy(self.config)
            self.canon = Instance(self.problem.canon_implementation, parent_config=config_tmp, is_groundtruth=True, suppress_override_message=True)

        if reference_instance is not None:
            self.reference = reference_instance
        elif self.config.use_canon_reference and self.problem.canon_implementation and not is_groundtruth:
            config_tmp = copy.copy(self.config)
            self.reference = Instance(self.problem.canon_implementation, parent_config=config_tmp, is_groundtruth=True, suppress_override_message=True)
        elif self.problem.all_correct_implementations and not is_groundtruth:
            self.reference = self.select_reference()
        # else:
        #     raise ValueError("No valid reference implementation found")

    def get_program_lines(self, input_id: typing.Union[None, int] = None) -> list[str]:
        from formhe.asp.synthesis.spec_generator import ASPSpecGenerator
        from formhe.asp.synthesis.AspVisitor import AspVisitor
        from formhe.fl.llm import ListInterpreter

        if input_id is None:
            ast = self.base_ast
        else:
            ast = self.asts[input_id]

        spec_generator = ASPSpecGenerator(self, self.config.extra_vars)
        trinity_spec = spec_generator.trinity_spec
        asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)
        asp_interpreter = ListInterpreter(self)

        lines = []
        for line in ast:
            try:
                node = asp_visitor.visit(line)
                lines.append(asp_interpreter.eval(node))
            except:
                lines.append(str(line))

        return [line for line in lines if line != ""]

    def get_program_str(self, input_id: typing.Union[None, int] = None) -> str:
        return "\n".join(self.get_program_lines(input_id))

    def get_control(self, *args, max_sols=0, i=0, instrumented=False, project=False, clingo_args: list = None):
        if clingo_args is None:
            clingo_args = []
        if project:
            clingo_args.append('--project')
        ctl = Control(clingo_args + [f'{max_sols}'], logger=lambda x, y: None)
        with clingo.ast.ProgramBuilder(ctl) as bld:
            if not instrumented:
                for stm in self.asts[i]:
                    bld.add(stm)
            else:
                for stm in self.instrumented_asts[i]:
                    bld.add(stm)
            if project:
                for p in self.problem.output_predicates_tuple:
                    clingo.ast.parse_string(f'#project {p[0]}/{p[1]}.', bld.add, logger=lambda x, y: None)
            for arg in args:
                clingo.ast.parse_string(arg, bld.add, logger=lambda x, y: None)
        runhelper.timer_start('grounding.time')
        try:
            ctl.ground([('base', [])])
        except:
            runhelper.timer_stop('grounding.time')
            raise InstanceGroundingException()
        runhelper.timer_stop('grounding.time')
        return ctl

    @cached_property
    def missing_models(self):
        missing = []
        for i in range(len(self.asts)):
            self.compute_models(0, i)
            self.canon.compute_models(0, i)
            missing.append(self.canon.models[i] - self.models[i])
        return missing

    @cached_property
    def extra_models(self):
        extra = []
        for i in range(len(self.asts)):
            self.compute_models(0, i)
            self.canon.compute_models(0, i)
            extra.append(self.models[i] - self.canon.models[i])
        return extra

    def compute_models(self, max_sols, i=0):
        if self.models[i] is not None:
            return
        else:
            self.models[i] = OrderedSet()

        if self.problem.optimization:
            ctl = self.get_control(max_sols=max_sols, i=i, project=True, clingo_args=['--opt-mode=optN'])
        else:
            ctl = self.get_control(max_sols=max_sols, i=i, project=True)

        def model_callback(m):
            if (not self.problem.optimization) or m.optimality_proven:
                tmp = m.symbols(shown=True)
                runhelper.tag_increment('answer.set.count')
                self.models[i].add(tuple(sorted([x for x in tmp])))

        runhelper.timer_start('answer.set.enum.time')
        ctl.solve(on_model=model_callback)
        runhelper.timer_stop('answer.set.enum.time')

    def verify(self, do_print=False) -> tuple[bool, list[bool]]:
        already_printed = False
        verification_status = []
        for i in range(len(self.instrumented_asts)):
            self.canon.compute_models(0, i)
            self.compute_models(len(self.canon.models) + 5 if not self.config.enum_all_answer_sets else 0, i)

        for i in range(len(self.instrumented_asts)):
            if len(self.models[i]) == len(self.canon.models[i]) == 0:
                verification_status.append(True)
                continue

            if len(self.models[i]) == 0 or (len(self.models[i]) == 1 and len(self.models[i][0]) == 0 and (len(self.canon.models[i]) != 1 or len(self.canon.models[i][0]) != 0)):
                if do_print and (not already_printed or not config.get().print_only_first_test_case):
                    print(f"Your program has failed test case {i+1}:\n")
                    print(re.sub('^', '\t', self.problem.inputs[i], flags=re.MULTILINE))
                    print(f'\nYour solution is overconstrained and does not produce any solutions for this input. Examples of correct answer sets:\n')
                    for m in self.canon.models[i][:config.get().model_feedback_n]:
                        print('\t' + ' '.join(map(str, m)))
                    print()

                already_printed = True
                verification_status.append(False)
                continue

            # if self.models[i] <= self.canon.models[i]:
            if len(self.extra_models[i]) == 0:
                verification_status.append(True)
                continue

            else:
                if do_print and (not already_printed or not config.get().print_only_first_test_case):
                    print(f"Your program has failed test case {i+1}:\n")
                    print(re.sub('^', '\t', self.problem.inputs[i], flags=re.MULTILINE))
                    print(f'\nYour solution is underconstrained and produces the following wrong models for this input:\n')
                    for model in (self.extra_models[i])[:config.get().model_feedback_n]:
                        if model:
                            print('\t' + ' '.join(map(str, model)))
                        else:
                            print('\t<empty answer set>')
                    print()

                already_printed = True
                verification_status.append(False)
                continue

        runhelper.log_any("verification.status", verification_status)

        return all(verification_status), verification_status

    def self_verify(self):
        from formhe.asp.synthesis.AspInterpreter import AspInterpreter

        if self.config.selfeval_fix is not None and self.config.selfeval_lines is not None:
            for i, (lines, fix) in enumerate(zip(self.config.selfeval_lines, self.config.selfeval_fix)):

                modified_instance = Instance(config.get().input_file, skips=lines, reference_instance=self.reference, canon_instance=self.canon)
                asp_interpreter = AspInterpreter(modified_instance)

                if not asp_interpreter.test(fix):
                    logger.error(f"Self-evaluation test {i} failed!")
                else:
                    logger.info(f"Self-evaluation test {i} successful")
        else:
            logger.warning('Self-evaluation fix test enabled, but selfeval lines or selfeval fix missing')

    def select_reference(self) -> 'Instance':
        from formhe.fl import line_matching

        runhelper.timer_start("reference.choice.time")
        references = []
        for correct_impl in self.problem.all_correct_implementations:
            config_tmp = copy.copy(self.config)
            references.append(Instance(correct_impl, parent_config=config_tmp, is_groundtruth=True, suppress_override_message=True))

        if not self.config.ignore_timestamps:
            references = [ref for ref in references if ref.config.timestamp < self.config.timestamp]

        min_cost = float("inf")
        min_ref = None
        for ref in references:
            try:
                cost = sum(map(lambda x: x[2], line_matching.pairings_with_cost(self, ref)))
                logger.debug("Distance to reference %s: %f", ref.filename, cost)
                if cost < min_cost:
                    min_cost = cost
                    min_ref = ref
            except Exception as e:
                logger.info("Failed to compute distance to reference %s because %s", ref.filename, str(e))

        if min_ref is None:
            logger.info("Using canon implementation")
            config_tmp = copy.copy(self.config)
            min_ref = Instance(self.problem.canon_implementation, parent_config=config_tmp, is_groundtruth=True, suppress_override_message=True)
        else:
            logger.info("Selecting reference %s with cost %f", min_ref.filename, min_cost)

        runhelper.timer_stop("reference.choice.time")
        return min_ref
