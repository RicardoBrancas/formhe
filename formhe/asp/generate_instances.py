import glob
from pathlib import Path

from asp.instance import Instance

for file in glob.glob('examples/*.lp'):
    example_name = Path(file).stem

    Path(f'generated_instances/{example_name}').mkdir(exist_ok=True)

    instance = Instance(file)

    test_lines = instance.integrity_constraints

    combos = []
    for i in range(len(test_lines)):
        list_copy = [str(stmt) for stmt in test_lines]
        list_copy.pop(i)
        combos.append((list_copy, str(test_lines[i])))

    for i, combo in enumerate(combos):

        with open(f'generated_instances/{example_name}/{i}.lp', 'w') as f:
            f.write(f'%formhe-groundtruth:../../examples/{example_name}.lp\n')
            for stmt in instance.others + instance.facts + instance.rules:
                f.write(str(stmt)+'\n')
            f.write('% selected out statement:\n')
            f.write(f'%{combo[1]}\n')
            f.write('\n'.join(combo[0]))
