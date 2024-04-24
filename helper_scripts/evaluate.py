#!/usr/bin/python

import argparse
import base64
import glob
import os
import random
import re
import sys
import warnings

from runhelper import Runner

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Util for benchmarking the SQUARES program synthesizer.')
parser.add_argument('-t', default=600, type=int, help='timeout')
parser.add_argument('-m', default=61440, type=int, help='memout')
# parser.add_argument('-n', default=1, type=int, help='number of times to run each instance')
parser.add_argument('-p', default=1, type=int, help='#processes')
parser.add_argument('--replace', action='store_true', help='replace previous run')
parser.add_argument('--sample', type=float)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--instances')
parser.add_argument('name', metavar='NAME', help="name of the result file")

args, other_args = parser.parse_known_args()
args_base64 = base64.b64encode(str(args).encode("utf-8")).decode("utf-8")


def process_data(instance: str, instance_data: dict, out_file: str):
    if os.path.isfile(out_file):
        with open(out_file) as f:
            log = f.read()

            if 'No problems found' in log:
                instance_data['feedback_type'] = 'Solution OK'
            elif "There is likely one or more bugs in the following statements" in log:
                instance_data['feedback_type'] = 'Detailed'
            elif 'Your solution is likely overconstrained' in log:
                instance_data['feedback_type'] = 'Basic - overconstrained'
            elif 'Your solution is likely underconstrained' in log:
                instance_data['feedback_type'] = 'Basic - underconstrained'
            elif 'Your solution is incorrect' in log:
                instance_data['feedback_type'] = 'Basic - incorrect'
            elif 'Error while parsing input file' in log:
                instance_data['feedback_type'] = 'Parsing Error'
            elif 'Error while grounding' in log:
                instance_data['feedback_type'] = 'Grounding Error'
            elif 'Synthesis Failed' in log:
                instance_data['feedback_type'] = 'Synthesis Failed'
            elif 'Solution found' in log:
                instance_data['feedback_type'] = 'Synthesis Success'
            elif log.strip() == '':
                instance_data['feedback_type'] = 'Empty'
            else:
                instance_data['feedback_type'] = 'Other'


if __name__ == '__main__':
    if not args.instances:
        instances = glob.glob(r'instances/**/*.lp', recursive=True)
    else:
        instances = []
        with open(args.instances) as inst_list:
            for inst in inst_list.readlines():
                instances += list(glob.glob(inst[:-1], recursive=True))

    if args.sample:
        instances = random.sample(instances, int(len(instances) * args.sample))

    if args.shuffle:
        random.shuffle(instances)

    runner = Runner('helper_scripts/runsolver', 'analysis/data/' + args.name + '.csv', timeout=args.t, memout=args.m, pool_size=args.p, token="6534160271:AAF2Wui-s86BgQxOhtoeNV5WzYFKDv1ahkk", chat_id="148166194")
    runner.register_instance_callback(process_data)

    for instance in instances:
        instance_name = re.search(r'.*?/(.*)\.lp', instance)[1]
        out_file = f'analysis/data/{args.name}/{instance_name}.log'
        command = [sys.executable, 'formhe/asp_integrated.py', instance, '--logging-level', 'INFO', '--no-stdin-instance', '--eval-params', args_base64] + other_args
        runner.schedule(instance_name, command, out_file)

    runner.wait()
