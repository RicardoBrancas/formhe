import glob
import pprint
import sys
from collections import Counter

import numpy as np

sys.argv.insert(1, "IGNORE")  # bypass requirement to pass an instance filename

import utils.clingo
from asp.synthesis.spec_generator import ASPSpecGenerator
from asp.synthesis.AspVisitor import AspVisitor
from formhe.exceptions.parser_exceptions import InstanceParseException, InstanceGroundingException
from utils import print_utils
from asp.instance import Instance
from formhe.trinity.DSL import ApplyNode, AtomNode
from scipy.optimize import linear_sum_assignment

extra_lines_counter = Counter()
total_cost_counter = Counter()


def compare_nodes(node_1, node_2):
    if isinstance(node_1, AtomNode) and isinstance(node_2, AtomNode):
        return 1 if node_1.production != node_2.production else 0

    elif isinstance(node_1, ApplyNode) and isinstance(node_2, ApplyNode):

        differences = 0

        if node_1.production != node_2.production:
            differences += 1

        for a, b in zip(node_1.children, node_2.children):
            differences += compare_nodes(a, b)

        differences += abs(len(node_1.children) - len(node_2.children))

        return differences

    elif isinstance(node_1, AtomNode) and isinstance(node_2, ApplyNode):
        return len(node_2.get_subtree())

    elif isinstance(node_1, ApplyNode) and isinstance(node_2, AtomNode):
        return len(node_1.get_subtree())

    elif node_1 is None and isinstance(node_2, ApplyNode) and node_2.production.name == "stmt":
        return 1 + len(node_2.children[1].children)

    elif isinstance(node_1, ApplyNode) and node_1.production.name == "stmt" and node_2 is None:
        return 1 + len(node_1.children[1].children)

    else:
        raise NotImplementedError()


def compare_lines(instance, fl_lines, correction_lines):
    spec_generator = ASPSpecGenerator(instance, instance.config.extra_vars, instance.constantCollector.predicates.items())
    trinity_spec = spec_generator.trinity_spec
    asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)

    try:
        fl_nodes = [asp_visitor.visit(l) for l in fl_lines]
    except (ValueError, NotImplementedError) as e:
        print("Error visiting faulty lines")
        return

    try:
        correction_nodes = [asp_visitor.visit(l) for l in correction_lines]
    except ValueError as e:
        print("Error visiting correction lines")
        return

    print([str(n) for n in fl_nodes])
    print([str(n) for n in correction_nodes])

    if len(fl_nodes) < len(correction_nodes):
        fl_nodes += [None] * (len(correction_nodes) - len(fl_nodes))

    if len(correction_nodes) < len(fl_nodes):
        correction_nodes += [None] * (len(fl_nodes) - len(correction_nodes))

    costs = np.zeros((len(fl_nodes), len(correction_nodes)))
    for i, node_1 in enumerate(fl_nodes):
        for j, node_2 in enumerate(correction_nodes):
            costs[i, j] = compare_nodes(node_1, node_2)

    row_ind, col_ind = linear_sum_assignment(costs)

    total_cost = 0
    for i, j in zip(row_ind, col_ind):
        print(fl_nodes[i], correction_nodes[j], costs[i, j])
        total_cost += costs[i, j]

    return total_cost


for instance_filename in glob.glob("instances/mooshak*/**/*.lp", recursive=True):
    print(instance_filename)

    try:
        instance = Instance(instance_filename)
    except InstanceParseException as e:
        continue
    except InstanceGroundingException as e:
        continue

    prog_lines = instance.get_program_lines()
    fl_lines = [line for (i, line) in enumerate(prog_lines) if i in instance.config.selfeval_lines[0]]
    fl_asts = utils.clingo.parse_string("\n".join(fl_lines))
    correct_lines = [line for (i, line) in enumerate(prog_lines) if i not in instance.config.selfeval_lines[0]]

    correction = instance.config.selfeval_fix[0]
    correction_asts = utils.clingo.parse_string(correction)
    correction_lines = instance.get_program_lines(ast_override=correction_asts)

    repaired_lines = correct_lines + correction_lines

    extra_lines_counter[len(repaired_lines) - len(prog_lines)] += 1

    print(fl_lines)
    print(correction_lines)

    total_cost_counter[compare_lines(instance, fl_asts, correction_asts)] += 1

    repaired_instance = Instance(ast=repaired_lines)

    print()
    print()
    print()
    print()

print("Needed extra lines:", print_utils.simplify(extra_lines_counter))
print("Total mutations:")
pprint.pp(print_utils.simplify(total_cost_counter), sort_dicts=True)
