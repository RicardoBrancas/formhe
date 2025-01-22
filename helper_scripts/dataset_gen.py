import glob
import logging
import os
import random
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from multiprocessing import Pool, RLock, current_process
from operator import itemgetter
from pathlib import Path

from argparse_dataclass import ArgumentParser
from datasets import Dataset
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as tqdmt

sys.argv.insert(1, "IGNORE")  # bypass requirement to pass an instance filename

from formhe.fl.llm import ListInterpreter
from formhe.asp.instance import Instance
from formhe.asp.synthesis.spec_generator import ASPSpecGenerator
from formhe.asp.synthesis.AspVisitor import AspVisitor
from formhe.exceptions.parser_exceptions import InstanceGroundingException
from formhe.trinity.ng_enumerator import NextGenEnumerator, PresetStatement
from formhe.utils.perm import PermutationGeneratorHelper
from formhe.utils import config
from formhe.asp.problem import Problem


def tqdm_wrapper(*args, **kwargs):
    if 'TELEGRAM_CHAT_ID' in os.environ and 'TELEGRAM_TOKEN' in os.environ:
        return tqdmt(*args, **kwargs, token=os.environ['TELEGRAM_TOKEN'], chat_id=os.environ['TELEGRAM_CHAT_ID'])
    else:
        return tqdm(*args, **kwargs)


@dataclass
class DataGenConfig:
    n_processes: int = 38
    seed: int = 45

    depths: list[int] = field(default_factory=lambda: [2, 3])
    cycles: int = 5
    n_gen_deletions: int = 30
    n_gen_mutations: dict[int, int] = field(default_factory=lambda: {1: 30, 2: 30, 3: 20, 4: 20, 5: 20, 6: 10, 7: 10, 8: 10})
    deletion_n_lines_probabilities: dict[int, float] = field(default_factory=lambda: {1: 0.4, 2: 0.4, 3: 0.2})
    max_percentage_incorrect_lines_threshold: float = 0.8
    max_errors_until_quit: int = 1000
    z3_timeout = 10000

    savefile_suffix: str = "8"
    instance_folder: str = "synthetic"
    instance_split_n = int = 500


def write_instances(results, datatgen_config: DataGenConfig):
    instance_counter = defaultdict(lambda: 0)

    for instance in results:
        out_filename = Path(f"{datatgen_config.instance_folder}_{datatgen_config.savefile_suffix}") / instance["problem"] / Path(instance["instance"]).stem / f"{instance_counter[instance['instance']]}.lp"
        out_filename.parent.mkdir(parents=True, exist_ok=True)

        instance_counter[instance['instance']] += 1

        with open(out_filename, "w") as f:
            f.write(f"%formhe-problem:{instance['problem']}\n")
            f.write(f"%formhe-timestamp:{instance['timestamp']}\n")
            f.write(f"%formhe-selfeval-lines:{' '.join(map(str, instance['fl']))}\n")
            f.write(f"%formhe-selfeval-fix:{' '.join(map(str, instance['correction']))}\n")
            f.write("\n")
            f.write(instance["incorrect_program"])


def create_instance_dict(short_name, incorrect_program, correct_program, n_mutations, problem, instance, datagen_config, correct_instances):
    incorrect_program_reduced = []
    reduce_mapping = {}
    reduce_offset = 0
    for i, line in enumerate(incorrect_program):
        if line == "":
            reduce_mapping[i] = None
            reduce_offset += 1
        else:
            incorrect_program_reduced.append(line)
            reduce_mapping[i] = i - reduce_offset
    if len(incorrect_program_reduced) == 0:
        return None
    incorrect_line_ids_pre = [i for i, b, in enumerate(zip(correct_program, incorrect_program)) if b[0] != b[1]]
    incorrect_line_ids = [reduce_mapping[i] for i, b, in enumerate(zip(correct_program, incorrect_program)) if b[0] != b[1] and b[1] != ""]
    fl = incorrect_line_ids + list(map(lambda x: x - reduce_offset, range(len(correct_program), len(incorrect_program))))
    correction_tmp = itemgetter(*incorrect_line_ids_pre)(correct_program) if len(incorrect_line_ids_pre) != 0 else ()
    correction = list(correction_tmp) if isinstance(correction_tmp, tuple) else [correction_tmp]
    if len(fl) / len(incorrect_program_reduced) >= datagen_config.max_percentage_incorrect_lines_threshold:
        return None
    for line_id in fl:
        assert line_id < len(incorrect_program_reduced)
    reference_instance = random.choice(correct_instances[problem.name])
    return {'instance': short_name,
            'problem': problem.name,
            'timestamp': instance.config.timestamp - 1 if instance.config.timestamp != 0 else 1,
            'correct_program_lines': tuple(correct_program),
            'correct_program': "\n".join(correct_program),
            'reference_program_lines': tuple(reference_instance),
            'reference_program': "\n".join(reference_instance),
            'incorrect_program_lines': tuple(incorrect_program_reduced),
            'incorrect_program': "\n".join(incorrect_program_reduced),
            'correction': tuple(correction),
            'n_mutations': n_mutations,
            'missing_lines': any([True for line in incorrect_program if line == ""]),
            'n_mising_lines': sum([1 for line in incorrect_program if line == ""]),
            'fl': tuple(fl),
            'line_scores': tuple([1.0 if i in fl else 0.0 for i in range(len(incorrect_program_reduced))])
            }


def process_correct_implementation(instance_file: str, problem: Problem, datagen_config: DataGenConfig, correct_instances):
    worker_id = current_process()._identity[0]
    short_name = instance_file.replace("correct_instances/", "")
    mutations_results = []
    t = tqdm(total=(sum(datagen_config.n_gen_mutations.values()) * len(datagen_config.depths) + datagen_config.n_gen_deletions) * datagen_config.cycles, desc=short_name, position=worker_id)

    for cycle_i in range(datagen_config.cycles):
        object.__setattr__(conf, "seed", hash(str(datagen_config.seed) + short_name + str(cycle_i)))

        try:
            instance = Instance(instance_file, shuffle=cycle_i != 0)
            instance_modified = Instance(instance_file, shuffle=instance.shuffling if cycle_i != 0 else False, skips=list(range(len(instance.base_ast))))

            for i in range(len(problem.inputs)):
                instance_modified.canon.compute_models(0, i=i)

            spec_generator = ASPSpecGenerator(instance, instance.config.extra_vars)
            trinity_spec = spec_generator.trinity_spec
            asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)
            asp_interpreter = ListInterpreter(instance_modified)

            correct_program = instance.get_program_lines()
            semi_bound_statements = []
            for rule in instance_modified.skipped_rules:
                node = asp_visitor.visit(rule)
                semi_bound_statements.append(PresetStatement(node.children[0], node.children[1].children))

        except Exception as ex:
            print("Skipping", short_name, "because", str(ex))
            return []

        for i in range(datagen_config.n_gen_deletions):
            to_delete_n = random.choices(list(datagen_config.deletion_n_lines_probabilities.keys()), list(datagen_config.deletion_n_lines_probabilities.values()))[0]
            lines_to_delete = random.sample(list(range(len(correct_program))), min(to_delete_n, len(correct_program) - 1))
            incorrect_program = [l if i not in lines_to_delete else "" for i, l in enumerate(correct_program)]
            if asp_interpreter.test("\n".join(incorrect_program)):
                continue
            d = create_instance_dict(short_name, incorrect_program, correct_program, -1, problem=problem, instance=instance_modified, datagen_config=datagen_config, correct_instances=correct_instances)
            if d is None:
                continue
            t.update()
            mutations_results.append(d)

        empty_statements = [PresetStatement(None, [])]

        for depth in datagen_config.depths:
            enumerator = NextGenEnumerator(trinity_spec, depth, semi_bound_statements=semi_bound_statements + empty_statements, additional_body_roots=1)

            enumerator.solver.set("timeout", datagen_config.z3_timeout)

            for n_mutations, mutations_per_program in datagen_config.n_gen_mutations.items():
                while enumerator.solver.num_scopes() >= 1:
                    enumerator.solver.pop()
                perm_helper = PermutationGeneratorHelper(str(datagen_config.seed) + short_name + str(cycle_i), len(enumerator.relaxation_vars()), n_mutations, enumerator, return_perms=True)
                mut_count = 0
                error_count = 0
                while mut_count < mutations_per_program and error_count <= datagen_config.max_errors_until_quit:
                    prog, perm = perm_helper.next()

                    if prog is None:
                        break

                    try:
                        asp_prog = asp_interpreter.eval(prog)
                        evaluation_result = asp_interpreter.test("\n".join(asp_prog))

                        if evaluation_result:
                            error_count += 1
                        else:
                            d = create_instance_dict(short_name, asp_prog, correct_program, n_mutations, problem=problem, instance=instance_modified, datagen_config=datagen_config, correct_instances=correct_instances)
                            if d is None:
                                error_count += 1
                                continue
                            mut_count += 1
                            t.update()
                            mutations_results.append(d)
                    except InstanceGroundingException:
                        error_count += 1
                        pass
                    except AssertionError as e:
                        error_count += 1
                        traceback.print_exception(e)
                    except Exception as e:
                        error_count += 1
                        pass
                        # traceback.print_exception(e)

    return mutations_results


if __name__ == "__main__":
    parser = ArgumentParser(DataGenConfig)
    datagen_config, _ = parser.parse_known_args()

    getLogger("formhe").setLevel(logging.ERROR)
    getLogger("urllib3").setLevel(logging.ERROR)

    conf = config.get()
    object.__setattr__(conf, "allow_unsafe_vars", True)
    object.__setattr__(conf, "allow_not_generated_predicates", True)
    object.__setattr__(conf, "disable_head_empty_or_non_constant_constraint", True)
    object.__setattr__(conf, "disable_no_dont_care_in_head_constraint", True)
    object.__setattr__(conf, "disable_no_dont_care_in_head_constraint", True)
    object.__setattr__(conf, "enable_pool_operator", True)

    problem_files = glob.glob("problems/*.yaml")
    problems = [Problem.from_yaml_file(f) for f in problem_files]

    correct_instances = defaultdict(list)
    for problem in problems:
        for correct_instance_file in problem.all_correct_implementations:
            correct_instances[problem.name].append(Instance(correct_instance_file).get_program_lines())

    results = []
    tqdm.set_lock(RLock())
    with Pool(datagen_config.n_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), maxtasksperchild=1) as pool:
        futures = []
        t = tqdm_wrapper(total=0)
        for problem in problems:
            for correct_instance_file in problem.all_correct_implementations:
                t.total += 1
                futures.append(pool.apply_async(process_correct_implementation, (correct_instance_file, problem, datagen_config, correct_instances)))

        for future in futures:
            results += future.get()
            t.update()

    df = DataFrame(results)
    print(len(df))
    df = df.drop_duplicates()
    print(len(df))
    ds = Dataset.from_pandas(df)
    ds = ds.train_test_split(test_size=500, seed=42)
    print(ds)
    write_instances(ds["test"], datagen_config)
    ds["train"].save_to_disk(f"datasets/dataset_{datagen_config.savefile_suffix}")
