#!/usr/bin/env python

import argparse

from asp.instance import Instance
from sygus.sygus_visitor import SyGuSVisitor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('INPUT')

    parser.add_argument('--model-attempts', default=500, type=int)
    parser.add_argument('--gt-model-attempts', default=2, type=int)

    parser.add_argument('--max-cores', default=10, type=int)
    parser.add_argument('--max-models', default=2, type=int)

    parser.add_argument('--constrain-reflexive', action='store_true')
    parser.add_argument('--relax-pbe-constraints', action='store_true')

    args = parser.parse_args()

    instance = Instance(args.INPUT)

    instance.find_wrong_models(max_sols=args.model_attempts)
    instance.generate_correct_models(max_sols=args.gt_model_attempts)

    sygus = SyGuSVisitor(instance.cores, instance.answer_sets, relax_pbe_constraints=args.relax_pbe_constraints, constrain_reflexive=args.constrain_reflexive)
    sygus.solve(max_cores=args.max_cores, max_models=args.max_models)


if __name__ == '__main__':
    main()
