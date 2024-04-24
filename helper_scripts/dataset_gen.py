import glob
import logging
import os
import random
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path

from argparse_dataclass import ArgumentParser
from ordered_set import OrderedSet
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as tqdmt

sys.argv.insert(1, "IGNORE")  # bypass requirement to pass an instance filename

from fl.LLM import ListInterpreter
from formhe.asp.instance import Instance
from formhe.asp.synthesis.ASPSpecGenerator import ASPSpecGenerator
from formhe.asp.synthesis.AspVisitor import AspVisitor
from formhe.exceptions.parser_exceptions import InstanceGroundingException
from formhe.trinity.ng_enumerator import NextGenEnumerator, PresetStatement
from formhe.utils.perm import PermutationGeneratorHelper
from utils import config


def tqdm_wrapper(*args, **kwargs):
    if 'TELEGRAM_CHAT_ID' in os.environ and 'TELEGRAM_TOKEN' in os.environ:
        return tqdmt(*args, **kwargs, token=os.environ['TELEGRAM_TOKEN'], chat_id=os.environ['TELEGRAM_CHAT_ID'])
    else:
        return tqdm(*args, **kwargs)


@dataclass
class DataGenConfig:
    n_processes: int = 8
    seed: int = 42

    depth: int = 2
    cycles: int = 1
    n_gen_deletions: int = 5
    n_gen_mutations: dict[int, int] = field(default_factory=lambda: {1: 5, 2: 5, 3: 3, 4: 2})
    deletion_n_lines_probabilities: dict[int, float] = field(default_factory=lambda: {1: 0.8, 2: 0.2})
    max_percentage_incorrect_lines_threshold: float = 0.75
    max_errors_until_quit: int = 1000

    save_dataset: bool = False
    save_files: str = "../instances_synthetic_new"


headers = {
    'A': """%formhe-groundtruth:../../../mooshak/instances/1920_2.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLDIpLiBlKDIsMykuIGUoMyw0KS4gZSg0LDUpLiBlKDUsMSkuCmUoMSw2KS4gZSgyLDcpLiBlKDMsOCkuIGUoNCw5KSwgZSg1LDEwKS4KZSg2LDgpLiBlKDYsOSkuCmUoNyw5KS4gZSg3LDEwKS4KZSg4LDEwKS4=""",
    'B': """%formhe-groundtruth:../../../mooshak/instances/1920_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=""",
    'C': """%formhe-groundtruth:../../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=""",
    'D': """%formhe-groundtruth:../../../mooshak/instances/1819_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgyLCAzKS4KZSgzLCA0KS4KZSg0LCA1KS4KZSg1LCA2KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAxLgoKZSgxLCAzKS4KZSgyLCAzKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4=""",
    'E': """%formhe-groundtruth:../../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K
%formhe-instance-base64:dmVydGV4KGEpLiB2ZXJ0ZXgoYikuIHZlcnRleChjKS4gdmVydGV4KGUpLiB2ZXJ0ZXgoMSkuIHZlcnRleCgyKS4gdmVydGV4KDMpLiB2ZXJ0ZXgoNCkuIHZlcnRleCg1KS4KZWRnZShhLDEpLiBlZGdlKGEsMikuIGVkZ2UoYiwzKS4gZWRnZShjLDIpLiBlZGdlKGMsNCkuIGVkZ2UoZSwxKS4gZWRnZShlLDQpLiBlZGdlKGUsNSku"""
}


def create_instance_dict(short_name, incorrect_program, correct_program, n_mutations, datagen_config):
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
    return {'instance': short_name,
            'problem': short_name[11],
            'correct_program_lines': correct_program,
            'correct_program': "\n".join(correct_program),
            'incorrect_program_lines': incorrect_program_reduced,
            'incorrect_program': "\n".join(incorrect_program_reduced),
            'correction': correction,
            'n_mutations': n_mutations,
            'missing_lines': any([True for line in incorrect_program if line == ""]),
            'n_mising_lines': sum([1 for line in incorrect_program if line == ""]),
            'fl': fl,
            'line_scores': [1.0 if i in fl else 0.0 for i in range(len(incorrect_program_reduced))]
            }


def process_instance(instance_file: str, datagen_config: DataGenConfig):
    short_name = instance_file.replace("../correct_instances/", "")
    print("Generating mutations for", short_name)
    mutations_results = []
    t = tqdm(total=((sum(datagen_config.n_gen_mutations.values())) + datagen_config.n_gen_deletions) * datagen_config.cycles)
    t.set_description(short_name)
    for cycle_i in range(datagen_config.cycles):
        object.__setattr__(conf, "seed", str(datagen_config.seed) + short_name + str(cycle_i))

        try:
            instance = Instance(instance_file, shuffle=cycle_i != 0)
        except Exception as ex:
            print("Skipping", short_name)
            print(ex)
            continue

        predicates = instance.constantCollector.predicates
        spec_generator = ASPSpecGenerator(instance, instance.config.extra_vars, predicates.items())
        trinity_spec = spec_generator.trinity_spec
        asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)
        asp_interpreter = ListInterpreter(instance, predicates.keys())

        try:
            correct_program = instance.get_program_lines()
            semi_bound_statements = []
            for rule in instance.constantCollector.not_skipped:
                node = asp_visitor.visit(rule)
                semi_bound_statements.append(PresetStatement(node.children[0], node.children[1].children))
        except Exception as ex:
            print("Skipping", short_name)
            print(ex)
            continue

        for i in range(datagen_config.n_gen_deletions):
            to_delete_n = random.choices(list(datagen_config.deletion_n_lines_probabilities.keys()), list(datagen_config.deletion_n_lines_probabilities.values()))[0]
            lines_to_delete = random.sample(list(range(len(correct_program))), to_delete_n)
            incorrect_program = [l if i not in lines_to_delete else "" for i, l in enumerate(correct_program)]
            d = create_instance_dict(short_name, incorrect_program, correct_program, -1, datagen_config=datagen_config)
            if d is None:
                continue
            t.update()
            mutations_results.append(d)

        empty_statements = [PresetStatement(None, [])]

        enumerator = NextGenEnumerator(trinity_spec, datagen_config.depth,
                                       semi_bound_statements=semi_bound_statements + empty_statements, free_predicates=OrderedSet(predicates.keys()) - instance.constantCollector.predicates_generated,
                                       force_generate_predicates=instance.constantCollector.predicates_used - instance.constantCollector.predicates_generated,
                                       additional_body_roots=1)

        enumerator.solver.set("timeout", 10000)

        for n_mutations, mutations_per_program in datagen_config.n_gen_mutations.items():
            while enumerator.solver.num_scopes() >= 1:
                enumerator.solver.pop()
            perm_helper = PermutationGeneratorHelper(str(datagen_config.seed) + short_name + str(cycle_i), len(enumerator.relaxation_vars()), n_mutations, enumerator, return_perms=True)
            mut_count = 0
            j = 0
            while mut_count < mutations_per_program and j <= datagen_config.max_errors_until_quit:
                prog, perm = perm_helper.next()

                if prog is None:
                    break

                try:
                    asp_prog = asp_interpreter.eval(prog)
                    evaluation_result = asp_interpreter.test("\n".join(asp_prog))

                    if evaluation_result:
                        j += 1
                        pass
                    else:
                        d = create_instance_dict(short_name, asp_prog, correct_program, n_mutations, datagen_config=datagen_config)
                        if d is None:
                            continue
                        mut_count += 1
                        t.update()
                        mutations_results.append(d)
                except InstanceGroundingException:
                    j += 1
                    pass
                except AssertionError as e:
                    j += 1
                    traceback.print_exception(e)
                except Exception as e:
                    j += 1
                    pass
                    # traceback.print_exception(e)

    return mutations_results


def tuplify(e):
    if len(e) == 0:
        return tuple()
    else:
        return tuple(e)


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

    instances = glob.glob("../correct_instances/mooshak/**/*.lp")

    results = []
    with Pool(datagen_config.n_processes) as pool:

        t = tqdm_wrapper(pool.starmap(process_instance, [(instance, datagen_config) for instance in instances], chunksize=1), total=len(instances))
        for r in t:
            results += r

    results = [{k: v if not isinstance(v, list) else tuplify(v) for k, v in r.items()} for r in results]

    if datagen_config.save_dataset:
        df = DataFrame(results)
        df.to_feather("dataset_4.feather")

    if datagen_config.save_files:
        instance_counter = defaultdict(lambda: 0)

        for instance in results:
            out_filename = Path(datagen_config.save_files) / instance["problem"] / Path(instance["instance"]).stem / f"{instance_counter[instance['instance']]}.lp"
            out_filename.parent.mkdir(parents=True, exist_ok=True)

            instance_counter[instance['instance']] += 1

            with open(out_filename, "w") as f:
                f.write(headers[instance["problem"]])
                f.write(f"\n%formhe-selfeval-lines:{' '.join(map(str, instance['fl']))}\n")
                f.write(f"%formhe-selfeval-fix:{' '.join(map(str, instance['correction']))}\n")
                f.write("\n")
                f.write(instance["incorrect_program"])
