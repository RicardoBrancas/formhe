import copy
import dataclasses
import logging
import re
import typing
from functools import cached_property
from itertools import product
from pathlib import Path

import clingo.ast
import utils.clingo
from clingo import Control
from clingo.ast import ASTType
from ordered_set import OrderedSet

import runhelper
from formhe.asp.utils import Visitor, Instrumenter
from formhe.exceptions.parser_exceptions import InstanceParseException, InstanceGroundingException
from formhe.utils import config, iterutils

logger = logging.getLogger('formhe.asp.instance')


class Instance:

    def __init__(self, filename: str = None, ast: clingo.ast.AST = None, skips: list[int] = None, parent_config: config.Config = None, ground_truth_instance=None):
        if (filename is not None and ast is not None) or (filename is None and ast is None):
            raise ValueError("Need to supply exactly one of {filename, ast}.")

        if filename is not None:
            with open(filename) as f:
                self.raw_input = f.read() + '\n'.join(config.stdin_content)
        elif ast is not None:
            self.raw_input = '\n'.join(map(str, ast))

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

        for key, value in compact_matches.items():
            logger.info('Overriding local config %s: %s', key, value)
            field_found = False
            for field in dataclasses.fields(config.Config):
                if field.name == key:
                    field_found = True
                    break
            if not field_found:
                raise AttributeError(key)
            if field.type is list or typing.get_origin(field.type) is list:
                if field.metadata and 'type' in field.metadata and callable(field.metadata['type']):
                    object.__setattr__(self.config, key, [field.metadata['type'](v) for v in value.split(' ') if v])
                else:
                    object.__setattr__(self.config, key, [v for v in value.split(' ') if v])
            elif field.type is bool:
                try:
                    object.__setattr__(self.config, key, config.strtobool(value))
                except:
                    object.__setattr__(self.config, key, value)
            elif field.metadata and 'type' in field.metadata and callable(field.metadata['type']):
                try:
                    object.__setattr__(self.config, key, field.metadata['type'](value))
                except:
                    object.__setattr__(self.config, key, value)
            else:
                object.__setattr__(self.config, key, value)

        if not self.config.instance_base64:
            self.inputs = [self.raw_input]
        else:
            self.inputs = []
            for instance64 in self.config.instance_base64:
                self.inputs.append(self.raw_input + '\nformhe_definition_begin.\n' + instance64 + '\nformhe_definition_end.')

        self.domain_predicates = OrderedSet()
        if self.config.domain_predicates:
            for pred in self.config.domain_predicates:
                pred_name, pred_arity = pred.split('/')
                self.domain_predicates.add((pred_name, int(pred_arity)))
                for i, input in enumerate(self.inputs):
                    self.inputs[i] = input + f'\n#show {pred}.'

        constantCollector = Visitor()

        self.asts = []
        self.instrumented_asts = []
        try:
            self.base_ast = utils.clingo.parse_string(self.raw_input)
            self.base_ast = iterutils.drop_nones([constantCollector.visit(node) for node in self.base_ast])
            for input in self.inputs:
                ast = utils.clingo.parse_string(input)
                self.constantCollector = Visitor(skips)  # TODO only the last instance counts
                self.asts.append(iterutils.drop_nones([self.constantCollector.visit(copy.copy(node)) for node in ast]))
                self.instrumenter = Instrumenter()
                self.instrumented_asts.append(iterutils.drop_nones([self.instrumenter.visit(node) for node in ast]))
        except Exception as e:
            raise InstanceParseException()
            # raise e

        if ground_truth_instance is not None:
            self.ground_truth = ground_truth_instance
            self.has_gt = True
        elif self.config.groundtruth:
            self.ground_truth_file = Path(filename).resolve().parent / self.config.groundtruth
            self.ground_truth_file = self.ground_truth_file.resolve()
            config_tmp = copy.copy(self.config)
            object.__setattr__(config_tmp, 'groundtruth', None)
            self.ground_truth = Instance(self.ground_truth_file, parent_config=config_tmp)
            self.has_gt = True
        else:
            self.has_gt = False

        self.constants = self.constantCollector.constants.items
        self.definitions = self.constantCollector.definitions.items
        self.cores = [OrderedSet() for i in range(len(self.asts))]
        self.gt_cores = [OrderedSet() for i in range(len(self.asts))]
        self.gt_unsat_models = [OrderedSet() for i in range(len(self.asts))]
        self.models = [OrderedSet() for i in range(len(self.asts))]
        self.controls = [self.get_control(i=i) for i in range(len(self.asts))]

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
                for p in self.domain_predicates:
                    clingo.ast.parse_string(f'#project {p[0]}/{p[1]}.', bld.add)
            for arg in args:
                clingo.ast.parse_string(arg, bld.add)
        runhelper.timer_start('grounding.time')
        try:
            ctl.ground([('base', [])])
        except:
            runhelper.timer_stop('grounding.time')
            raise InstanceGroundingException()
        runhelper.timer_stop('grounding.time')
        return ctl

    # @lru_cache(config.get().model_cache_size)
    # def check_model(self, model: Iterable[Symbol], i=0) -> bool:
    #     control = self.get_control(*[':- not ' + str(atom) + '.' for atom in model], i=i)
    #     res = control.solve(assumptions=[(x, True) for x in model])
    #     # control = self.controls[i]
    #     # res = control.solve(assumptions=[(x, True) for x in model])
    #     if res.unsatisfiable:
    #         runhelper.tag_increment('unique.unsat')
    #         return False
    #     elif res.satisfiable:
    #         runhelper.tag_increment('unique.sat')
    #     else:
    #         raise RuntimeError()
    #     return True

    @cached_property
    def missing_models(self):
        missing = []
        for i in range(len(self.asts)):
            self.compute_models(0, i)
            self.ground_truth.compute_models(0, i)
            missing.append(self.ground_truth.models[i] - self.models[i])
        return missing

    @cached_property
    def extra_models(self):
        extra = []
        for i in range(len(self.asts)):
            self.compute_models(0, i)
            self.ground_truth.compute_models(0, i)
            extra.append(self.models[i] - self.ground_truth.models[i])
        return extra

    def compute_models(self, max_sols, i=0):
        if self.config.optimization_problem:
            ctl = self.get_control(max_sols=max_sols, i=i, project=True, clingo_args=['--opt-mode=optN'])
        else:
            ctl = self.get_control(max_sols=max_sols, i=i, project=True)

        def model_callback(m):
            if (not self.config.optimization_problem) or m.optimality_proven:
                tmp = m.symbols(shown=True)
                runhelper.tag_increment('answer.set.count')
                self.models[i].add(tuple(sorted([x for x in tmp])))

        ctl.solve(on_model=model_callback)

    def mcs_negated_query(self, model, relaxed):
        if not relaxed:
            negated_query = ' '.join([':- not ' + str(x) + '.' for x in model if x])
        else:
            negated_query = ' '.join(['' + str(x) + '.' for x in model if x])
        return negated_query

    def mcs_query(self, model):
        query = ':- ' + (', '.join([str(x) for x in model if x])) + '.'
        return query

    def all_mcs(self, model, relaxed=False, i=0, positive=False):
        logger.info(f'Iterating all {"relaxed " if relaxed else ""}MCSs')

        instrumenter_vars = self.instrumenter.relaxation_functions
        generate_vars = '0 { ' + '; '.join(map(str, instrumenter_vars)) + ' } ' + str(len(instrumenter_vars)) + '.'
        clause_r = ''
        mcs_blocks = OrderedSet()
        mcss = OrderedSet()

        if positive:
            negated_query = ' '.join([self.mcs_query(m) for m in model])
        else:
            negated_query = self.mcs_negated_query(model, relaxed)

        logger.info('Transformed query: %s', negated_query)

        unsatisfied_vars = None
        unsat = True
        while True:
            ctl = self.get_control(generate_vars, clause_r, negated_query, *mcs_blocks, max_sols=1, instrumented=True, i=i)

            with ctl.solve(yield_=True) as handle:
                res = handle.get()

                if res.satisfiable:
                    unsat = False
                    m = handle.model()
                    unsatisfied_vars = []
                    satisfied_vars = []

                    for var in instrumenter_vars:
                        if not m.contains(var):
                            unsatisfied_vars.append(var)
                        else:
                            satisfied_vars.append(var)

                    unsatisfied_vars = frozenset(unsatisfied_vars)
                    satisfied_vars = frozenset(satisfied_vars)

                    # print(clause_r if clause_r else generate_vars)
                    # for mcs_block in mcs_blocks:
                    #     print(mcs_block)
                    # print('SAT')
                    # print('satisfied:', list(map(str, satisfied_vars)))
                    # print('unsatisfied:', list(map(str, unsatisfied_vars)))
                    # print('\n\n\n')

                    clause_r = f'{len(satisfied_vars) + 1} {{ {"; ".join(map(str, instrumenter_vars))} }} {len(instrumenter_vars)}.'

                elif res.unsatisfiable:
                    # print(clause_r if clause_r else generate_vars)
                    # for mcs_block in mcs_blocks:
                    #     print(mcs_block)
                    # print('UNSAT')
                    # print('\n\n\n')

                    if not unsat:
                        if not any(map(lambda mcs: mcs.issubset(unsatisfied_vars), mcss)):
                            yield frozenset(self.instrumenter.relaxations_function_map[var] for var in unsatisfied_vars)
                            mcss.append(unsatisfied_vars)
                            logger.info(' '.join(str(x) for x in unsatisfied_vars))
                            clause_r = ''
                            mcs_blocks.append(':- ' + ','.join(map(lambda x: 'not ' + str(x), unsatisfied_vars)) + '.')
                            unsat = True
                        else:
                            raise NotImplementedError()
                            # clause_r = ''
                            # mcs_blocks.append(':- ' + ','.join(map(lambda x: 'not ' + str(x), unsatisfied_vars)) + '.')
                            # unsat = True
                    else:
                        return

                else:
                    raise RuntimeError()

    def all_weak_mcs(self, model, relaxed=False, i=0, positive=False):
        logger.info(f'Iterating all strong {"relaxed " if relaxed else ""}MCSs')

        instrumenter_vars = self.instrumenter.relaxation_functions
        mcs_blocks = OrderedSet()
        mcss = OrderedSet()

        if positive:
            negated_query = ' '.join([self.mcs_query(m) for m in model])
        else:
            negated_query = self.mcs_negated_query(model, relaxed)

        logger.info('Transformed query: %s', negated_query)

        while True:
            untested_vars = OrderedSet(instrumenter_vars)
            unsatisfied_vars = OrderedSet()
            satisfied_vars = OrderedSet()
            unsat = True
            while untested_vars:
                var = untested_vars[0]
                untested_vars = untested_vars - {var}
                clause_s = ' '.join(map(lambda x: f'{x}.', satisfied_vars.union({var})))
                keep_unsats = ' '.join(map(lambda x: f'-{x}.', unsatisfied_vars))
                clause_choice = '0 { ' + '; '.join(map(str,  OrderedSet(instrumenter_vars) - satisfied_vars.union({var}))) + ' }.'
                print(clause_s)
                print(keep_unsats)
                print(negated_query)
                for mcs_block in mcs_blocks:
                    print(mcs_block)
                print()
                print()
                print()
                # ctl = self.get_control(clause_s, clause_choice, negated_query, *mcs_blocks, max_sols=1, instrumented=True, i=i)
                ctl = self.get_control(clause_s, keep_unsats, negated_query, *mcs_blocks, max_sols=1, instrumented=True, i=i)

                with ctl.solve(yield_=True) as handle:
                    res = handle.get()

                    if res.satisfiable:
                        unsat = False
                        satisfied_vars.add(var)
                    elif res.unsatisfiable:
                        unsatisfied_vars.add(var)
                    else:
                        raise RuntimeError()

            if not unsat:
                yield frozenset(self.instrumenter.relaxations_function_map[var] for var in unsatisfied_vars)
                mcss.append(frozenset(unsatisfied_vars))
                logger.info(' '.join(str(x) for x in unsatisfied_vars))
                clause_r = ''
                mcs_block = ''
                mcs_blocks.append('1 { ' + '; '.join(map(str, unsatisfied_vars)) + ' }.')
                unsat = True
            else:
                return

    def all_min_mcs(self, model, relaxed=False, i=0, positive=False):
        logger.info(f'Iterating all {"relaxed " if relaxed else ""}min MCSs')

        instrumenter_vars = self.instrumenter.relaxation_functions
        generate_vars = '0 { ' + '; '.join(map(str, instrumenter_vars)) + ' } ' + str(len(instrumenter_vars)) + '.'

        if positive:
            negated_query = ' '.join([self.mcs_query(m) for m in model])
        else:
            negated_query = self.mcs_negated_query(model, relaxed)

        logger.info('Transformed query: %s', negated_query)

        ctl = self.get_control(generate_vars, negated_query, '#maximize{ 1, I : _instrumenter(I) }.', "#project _instrumenter/1.", instrumented=True, clingo_args=['--project', '--opt-mode=optN'], i=i)
        with ctl.solve(yield_=True) as handle:
            for m in handle:
                if m.optimality_proven:
                    unsatisfied_vars = []
                    for var in instrumenter_vars:
                        if not m.contains(var):
                            unsatisfied_vars.append(var)
                    logger.info(' '.join(str(x) for x in unsatisfied_vars))
                    yield frozenset(self.instrumenter.relaxations_function_map[var] for var in unsatisfied_vars)
        return

    def all_mfl(self, model, relaxed=False, i=0, positive=False):
        ctl = self.get_control(max_sols=1, i=i)

        supports = []
        support_vars = []

        for symbolic_atom in ctl.symbolic_atoms:
            symbol = symbolic_atom.symbol
            supports.append(f'{symbol} :- not _support({symbol}).')
            support_vars.append(clingo.Function(f"_support", [symbol]))

        instrumenter_vars = self.instrumenter.relaxation_functions
        n = len(instrumenter_vars + support_vars)
        generate_vars = '0 { ' + '; '.join(map(str, instrumenter_vars + support_vars)) + ' } ' + str(n) + '.'
        clause_r = ''
        mcs_blocks = OrderedSet()
        mcss = OrderedSet()

        if positive:
            negated_query = ' '.join([self.mcs_query(m) for m in model])
        else:
            negated_query = self.mcs_negated_query(model, relaxed)

        logger.info('Transformed query: %s', negated_query)

        unsatisfied_vars = None
        unsat = True
        while True:
            ctl = self.get_control(generate_vars, clause_r, negated_query, ' '.join(supports), *mcs_blocks, max_sols=1, instrumented=True, i=i)

            with ctl.solve(yield_=True) as handle:
                res = handle.get()

                if res.satisfiable:
                    unsat = False
                    m = handle.model()
                    unsatisfied_vars = []
                    satisfied_vars = []

                    for var in instrumenter_vars + support_vars:
                        if not m.contains(var):
                            unsatisfied_vars.append(var)
                        else:
                            satisfied_vars.append(var)

                    unsatisfied_vars = frozenset(unsatisfied_vars)
                    satisfied_vars = frozenset(satisfied_vars)

                    # print(clause_r if clause_r else generate_vars)
                    # for mcs_block in mcs_blocks:
                    #     print(mcs_block)
                    # print('SAT')
                    # print('satisfied:', list(map(str, satisfied_vars)))
                    # print('unsatisfied:', list(map(str, unsatisfied_vars)))
                    # print('\n\n\n')

                    clause_r = f'{len(satisfied_vars) + 1} {{ {"; ".join(map(str, instrumenter_vars + support_vars))} }} {n}.'

                elif res.unsatisfiable:
                    # print(clause_r if clause_r else generate_vars)
                    # for mcs_block in mcs_blocks:
                    #     print(mcs_block)
                    # print('UNSAT')
                    # print('\n\n\n')

                    if not unsat:
                        if not any(map(lambda mcs: mcs.issubset(unsatisfied_vars), mcss)):
                            unsatisfied_rules = [self.instrumenter.relaxations_function_map[var] for var in unsatisfied_vars if var in instrumenter_vars]
                            for var in unsatisfied_vars:
                                if var in support_vars:
                                    predicate = var.arguments[0].name
                                    for rule_i, rule in enumerate(self.instrumenter.rules):
                                        if rule.ast_type == ASTType.Rule and \
                                                rule.head and \
                                                rule.head.ast_type == ASTType.Literal and \
                                                rule.head.atom.ast_type == ASTType.SymbolicAtom and \
                                                rule.head.atom.symbol.name == predicate:
                                            unsatisfied_rules.append(rule_i)
                                        elif rule.ast_type == ASTType.Rule and \
                                                rule.head and \
                                                rule.head.ast_type == ASTType.Aggregate:
                                            for element in rule.head.elements:
                                                if element.ast_type == ASTType.ConditionalLiteral and \
                                                        element.literal.ast_type == ASTType.Literal and \
                                                        element.literal.atom.ast_type == ASTType.SymbolicAtom and \
                                                        element.literal.atom.symbol.name == predicate:
                                                    unsatisfied_rules.append(rule_i)
                            yield frozenset(unsatisfied_rules)
                            mcss.append(unsatisfied_vars)
                            logger.info(' '.join(str(x) for x in unsatisfied_vars))
                            clause_r = ''
                            mcs_blocks.append(':- ' + ','.join(map(lambda x: 'not ' + str(x), unsatisfied_vars)) + '.')
                            unsat = True
                        else:
                            raise NotImplementedError()
                            # clause_r = ''
                            # mcs_blocks.append(':- ' + ','.join(map(lambda x: 'not ' + str(x), unsatisfied_vars)) + '.')
                            # unsat = True
                    else:
                        return

                else:
                    raise RuntimeError()

    def all_min_mfl(self, model, relaxed=False, i=0, positive=False):
        ctl = self.get_control(max_sols=1, i=i)

        supports = []
        support_vars = []

        for symbolic_atom in ctl.symbolic_atoms:
            symbol = symbolic_atom.symbol
            supports.append(f'{symbol} :- not _support({symbol}).')
            support_vars.append(clingo.Function(f"_support", [symbol]))

        instrumenter_vars = self.instrumenter.relaxation_functions
        n = len(instrumenter_vars + support_vars)
        generate_vars = '0 { ' + '; '.join(map(str, instrumenter_vars + support_vars)) + ' } ' + str(n) + '.'

        if positive:
            negated_query = ' '.join([self.mcs_query(m) for m in model])
        else:
            negated_query = self.mcs_negated_query(model, relaxed)

        logger.info('Transformed query: %s', negated_query)

        ctl = self.get_control(generate_vars, negated_query, ' '.join(supports), '#maximize{ 1, I, A : _instrumenter(I), _support(A) }.', "#project _instrumenter/1.", "#project _support/1.",
                               instrumented=True, clingo_args=['--project', '--opt-mode=optN'], i=i)
        with ctl.solve(yield_=True) as handle:
            for m in handle:
                if m.optimality_proven:
                    # print(m.symbols(atoms=True))
                    unsatisfied_vars = []
                    for var in instrumenter_vars:
                        if not m.contains(var):
                            unsatisfied_vars.append(var)
                    supported_vars = []
                    rules_from_supports = []
                    for var in support_vars:
                        if not m.contains(var):
                            supported_vars.append(var)
                            predicate = var.arguments[0].name
                            for rule_i, rule in enumerate(self.instrumenter.rules):
                                if rule.ast_type == ASTType.Rule and \
                                        rule.head and \
                                        rule.head.ast_type == ASTType.Literal and \
                                        rule.head.atom.ast_type == ASTType.SymbolicAtom and \
                                        rule.head.atom.symbol.name == predicate:
                                    rules_from_supports.append(rule_i)
                    logger.info(' '.join(str(x) for x in unsatisfied_vars))
                    logger.info(' '.join(str(x) for x in supported_vars))
                    yield frozenset([self.instrumenter.relaxations_function_map[var] for var in unsatisfied_vars] + rules_from_supports)
        return

    def bag_of_nodes(self):
        from formhe.asp.synthesis.AspVisitor import AspVisitor, bag_nodes
        from formhe.asp.synthesis.ASPSpecGenerator import ASPSpecGenerator
        impl_lines = []
        impl_orig = []
        impl_nodes = []

        spec_generator = ASPSpecGenerator(self, 0, self.constantCollector.predicates.items())
        trinity_spec = spec_generator.trinity_spec
        asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars, True, domain_predicates=self.domain_predicates)
        for rule in self.base_ast:
            if rule.ast_type != clingo.ast.ASTType.Rule:
                continue
            node = asp_visitor.visit(rule)
            if isinstance(node, tuple):
                node = asp_visitor.builder.make_apply('stmt', [node[0] if node[0] else asp_visitor.builder.make_apply('empty', []), asp_visitor.builder.make_apply('and_', node[1])])
            impl_lines.append(bag_nodes(node))
            impl_orig.append(rule)
            impl_nodes.append(node)

        return impl_lines, impl_orig, impl_nodes

    def line_pairings(self, reference=None):
        from multiset import Multiset
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        from formhe.asp.synthesis.AspVisitor import AspVisitor
        from formhe.asp.synthesis.ASPSpecGenerator import ASPSpecGenerator

        if reference is None:
            reference = self.ground_truth

        try:
            impl_lines, impl_orig, impl_nodes = Instance.bag_of_nodes(self)
            reference_impl_lines, reference_impl_orig, reference_impl_nodes = Instance.bag_of_nodes(reference)

            spec_generator = ASPSpecGenerator(self, 0, self.constantCollector.predicates.items())
            trinity_spec = spec_generator.trinity_spec
            asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars, True, domain_predicates=self.domain_predicates)
            if len(reference_impl_lines) < len(impl_lines):
                n = len(impl_lines) - len(reference_impl_lines)
                reference_impl_lines += [Multiset(['empty']) for i in range(n)]
                reference_impl_orig += [None for i in range(n)]
                reference_impl_nodes += [asp_visitor.builder.make_apply('empty', []) for i in range(n)]
            if len(impl_lines) < len(reference_impl_lines):
                n = len(reference_impl_lines) - len(impl_lines)
                impl_lines += [Multiset(['empty']) for i in range(n)]
                impl_orig += [None for i in range(n)]
                impl_nodes += [asp_visitor.builder.make_apply('empty', []) for i in range(n)]

            costs = np.zeros((len(reference_impl_lines), len(impl_lines)))
            for (i, a), (j, b) in product(enumerate(reference_impl_lines), enumerate(impl_lines)):
                # costs[i, j] = len(a.symmetric_difference(b)) / (len(a) + len(b)) if len(a) + len(b) != 0 else 0
                costs[i, j] = len(a.symmetric_difference(b))

            row_ind, col_ind = linear_sum_assignment(costs)

            logger.debug('Printing line pairings followed by pairing cost')
            pairings_with_cost = []
            for a, b in zip(row_ind, col_ind):
                logger.debug(reference_impl_orig[a])
                logger.debug(reference_impl_nodes[a])
                logger.debug(reference_impl_lines[a])
                logger.debug(impl_orig[b])
                logger.debug(impl_nodes[b])
                logger.debug(impl_lines[b])
                logger.debug(costs[a, b])
                pairings_with_cost.append((a, b, costs[a, b]))

            runhelper.log_any('pairings', pairings_with_cost)

            return pairings_with_cost
        except Exception as e:
            print(e)
            logger.error('Exception while trying to compute line pairings')
            return None

    def print_answer_sets(self):
        for a in sorted(self.answer_sets, key=len):
            print(a)

    def print_cores(self):
        for c in sorted(self.cores, key=len):
            print(c)

    def print_gt_cores(self):
        for c in sorted(self.gt_cores, key=len):
            print(c)
