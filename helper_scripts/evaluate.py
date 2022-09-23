#!/usr/bin/python

import argparse
import base64
import csv
import glob
import os
import pathlib
import random
import re
import subprocess
import warnings
from multiprocessing import Pool

from ordered_set import OrderedSet

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Util for benchmarking the SQUARES program synthesizer.')
parser.add_argument('-t', default=600, type=int, help='timeout')
parser.add_argument('-m', default=61440, type=int, help='memout')
parser.add_argument('-n', default=1, type=int, help='number of times to run each instance')
parser.add_argument('-p', default=1, type=int, help='#processes')
parser.add_argument('--append', action='store_true', help='append to file')
parser.add_argument('--resume', action='store_true', help='resume previous run')
parser.add_argument('--replace', action='store_true', help='replace previous run')
parser.add_argument('--sample', type=float)
parser.add_argument('--instances')
parser.add_argument('name', metavar='NAME', help="name of the result file")

args, other_args = parser.parse_known_args()
args_base64 = base64.b64encode(str(args).encode("utf-8")).decode("utf-8")

columns = OrderedSet()
rows = []


def test_file(filename: str, run: str = ''):
    test_name = re.search(r'buggy_instances/(.*)\.lp', filename)[1]
    out_file = f'analysis/data/{args.name}/{test_name}{run}.log'
    pathlib.Path(os.path.dirname(out_file)).mkdir(parents=True, exist_ok=True)

    command = ['helper_scripts/runsolver', '-W', str(args.t), '--rss-swap-limit', str(args.m), '-d', '5', '-o', out_file, 'python', 'formhe/asp_integrated.py', filename, '--logging-level', 'INFO', '--eval-params', args_base64]

    command += other_args

    print(' '.join(command))
    p = subprocess.run(command, capture_output=True, encoding='utf8')

    instance_data = {'instance': test_name,
                     'real': float(re.search('Real time \(s\): (.*)', p.stdout)[1]), 'cpu': float(re.search('CPU time \(s\): (.*)', p.stdout)[1]),
                     'ram': int(re.search('Max. memory \(cumulated for all children\) \(KiB\): (.*)', p.stdout)[1]),
                     'timeout': re.search('Maximum wall clock time exceeded: sending SIGTERM then SIGKILL', p.stdout) is not None,
                     'memout': re.search('Maximum memory exceeded: sending SIGTERM then SIGKILL', p.stdout) is not None}

    try:
        instance_data['status'] = re.search('Child status: (.*)', p.stdout)[1]
    except:
        instance_data['status'] = None if instance_data['timeout'] or instance_data['memout'] else 0

    if not instance_data['timeout'] and not instance_data['memout']:
        with open(out_file) as f:
            log = f.read()
            for tag, value in re.findall(r'perf\.(.*)=(.*)', log):
                instance_data[tag] = value

    if not all(map(lambda k: k in columns, instance_data.keys())):
        print('New tag found. Re-printing data.')
        for key in instance_data.keys():
            if key not in columns:
                columns.add(key)

        with open('analysis/data/' + args.name + '.csv_', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

        os.remove('analysis/data/' + args.name + '.csv')
        os.rename('analysis/data/' + args.name + '.csv_', 'analysis/data/' + args.name + '.csv')

    row = tuple(instance_data.get(key, None) for key in columns)
    rows.append(row)

    with open('analysis/data/' + args.name + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()


if __name__ == '__main__':
    if not args.append and not args.resume:
        if not args.replace:
            os.mkdir(f'analysis/data/{args.name}')
        with open('analysis/data/' + args.name + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('name', 'timeout', 'real', 'cpu', 'ram', 'process', 'status', 'memout'))
            f.flush()

    if not args.instances:
        instances = glob.glob('buggy_instances/**/*.lp', recursive=True)
    else:
        instances = []
        with open(args.instances) as inst_list:
            for inst in inst_list.readlines():
                instances += list(glob.glob(inst[:-1], recursive=True))

    if args.sample:
        instances = random.sample(instances, int(len(instances) * args.sample))

    if args.resume:
        with open('analysis/data/' + args.name + '.csv', 'r') as f:
            reader = csv.reader(f)
            existing_instances = []
            for row in reader:
                existing_instances.append('tests/' + row[0] + '.yaml')
                print('Skipping', 'tests/' + row[0] + '.yaml')

        instances = filter(lambda x: x not in existing_instances, instances)

    if args.p == 1:
        for i in range(args.n):
            for file in instances:
                test_file(file, f'_{i}')
    else:
        with Pool(processes=args.p) as pool:
            for i in range(args.n):
                pool.map(test_file, instances, chunksize=1)
