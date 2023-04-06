#!/usr/bin/env python

import argparse
import logging
from itertools import islice

from ordered_set import OrderedSet

from formhe.asp.synthesis.ASPSpecGenerator import ASPSpecGenerator
from formhe.asp.instance import Instance
from formhe.sygus.sygus_visitor import SyGuSVisitor

logger = logging.getLogger('formhe.asp.integrated')


def mk_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('INPUT', help='A ``.lp`` file containing an incomplete ASP program.')

    parser.add_argument('--model-attempts', default=2000, type=int,
                        help='Number of models generated from the incomplete program')
    parser.add_argument('--gt-model-attempts', default=5, type=int,
                        help='Number of models generated from the complete program')

    parser.add_argument('--max-cores', default=20, type=int,
                        help='Maximum number of UNSAT cores used in the SyGuS sprecification')
    parser.add_argument('--max-models', default=5, type=int,
                        help='Maximum number of correct models used in the SyGuS sprecification')

    parser.add_argument('--constrain-reflexive', action='store_true')
    parser.add_argument('--relax-pbe-constraints', action='store_true')

    parser.add_argument('--find-minimum', action='store_true', help='Find a minimum MCS instead of a minimal one.')

    parser.add_argument("--query", action="extend", nargs="+", type=str,
                        help='A set of atoms which should appear in some model but do not')

    debug_group = parser.add_argument_group(title='Debug Options')

    debug_group.add_argument('--skip-cores', default=0, type=int)

    return parser


def main():
    parser = mk_argument_parser()

    args = parser.parse_args()

    instance = Instance(args.INPUT)

    print('Instrumented program:\n')

    for node in instance.instrumented_ast:
        print(node)

    print()

    if not args.query and instance.mcs_query:
        print('Reading query from instance file...')
        print(instance.mcs_query)
        query = instance.mcs_query
    else:
        query = ' '.join(args.query)

    unsats_union = OrderedSet()

    for unsat in islice(instance.all_mcs(query), 100):
        for var in unsat:
            unsats_union.add(var)

    print(unsats_union)

    instance = Instance(args.INPUT, skips=unsats_union)

    spec_generator = ASPSpecGenerator(instance, 2)
    trinity_spec = spec_generator.trinity_spec

    print(instance.constantCollector.skipped)

    print()
    print('Modified program:')
    print()

    for node in instance.ast:
        print(node)

    print()

    instance.find_wrong_models(max_sols=args.model_attempts)
    instance.generate_correct_models(max_sols=args.gt_model_attempts)

    sygus = SyGuSVisitor(instance, instance.cores, instance.answer_sets,
                         relax_pbe_constraints=args.relax_pbe_constraints, constrain_reflexive=args.constrain_reflexive,
                         skip_cores=args.skip_cores)
    sygus.solve(max_cores=args.max_cores, max_models=args.max_models, skip_cores=args.skip_cores)


if __name__ == '__main__':
    main()
