#!/usr/bin/env python

import argparse

from formhe.asp.instance import Instance
from formhe.sygus.sygus_visitor import SyGuSVisitor


def mk_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('INPUT', help='A ``.lp`` file containing an incomplete ASP program.')

    parser.add_argument('--find-minimum', action='store_true', help='Find a minimum MCS instead of a minimal one.')

    parser.add_argument("--query", action="extend", nargs="+", type=str, help='A set of atoms which should appear in some model but do not')

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

    unsats = instance.find_mcs(query, args.find_minimum)

    print(unsats)


if __name__ == '__main__':
    main()
