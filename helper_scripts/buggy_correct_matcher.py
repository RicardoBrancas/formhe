import glob
import logging
import math
from pathlib import Path

from formhe.exceptions.parser_exceptions import InstanceParseException, InstanceGroundingException

from formhe.asp.instance import Instance

logging.getLogger().setLevel(logging.ERROR)

for file in glob.glob('buggy_instances/**/*.lp', recursive=True):
    file_path = Path(file)
    directory = file_path.parent
    problem, user, stamp = file_path.stem.split('_')

    print(directory, problem, user, stamp)

    minimum = None
    minimum_score = math.inf
    for other_file in glob.glob(str(str(directory).replace('buggy_instances', 'correct_instances') + '/' + (problem + '_*.lp'))):
        other_file_path = Path(other_file)
        _, other_user, other_stamp = other_file_path.stem.split('_')

        if other_user == user:
            continue

        # print('\t', directory, problem, other_user, other_stamp)

        try:
            instance = Instance(file)
        except (InstanceParseException, InstanceGroundingException):
            continue

        other_instance = Instance(other_file)

        line_pairings = instance.line_pairings(other_instance)

        if line_pairings:
            score_sum = sum(map(lambda x: x[2], line_pairings))

            if score_sum < minimum_score:
                minimum = other_file
                minimum_score = score_sum

    print('\t', minimum)
    print('\t', minimum_score)
