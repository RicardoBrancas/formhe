import glob
from ast import literal_eval
from collections import defaultdict

count = 0

two_grams = defaultdict(lambda: defaultdict(lambda: 0))


def update_two_gram_(parent: str, child: str):
    if parent.startswith('.') and child.startswith('.'):
        return
    parent = parent.removeprefix('.')
    child = child.removeprefix('.')
    two_grams[parent][child] += 1


def update_two_gram(prog: dict, parent):
    if isinstance(prog, dict):
        assert len(prog.keys()) == 1
        child = next(iter(prog.keys()))
        update_two_gram_(str(parent), str(child))
        for child_child in prog[child]:
            update_two_gram(child_child, child)
    else:
        update_two_gram_(str(parent), str(prog))


for filename in glob.glob("analysis/data/1007/**/*.log", recursive=True):
    with open(filename) as f:
        for line in f.readlines():
            if not line.startswith("["):
                continue
            count = count + 1
            program = literal_eval(line)
            for stmt in program:
                head = stmt[0]
                bodies = stmt[1:]
                if head != 0:
                    update_two_gram(head, '.root')
                    # print(head)
                for body in bodies:
                    if body != 0:
                        update_two_gram(body, '.root')
                        # print(body)

print(count)
print({k: {k2: v2 for k2, v2 in v.items()} for k, v in two_grams.items()})

print('word1,word2,count')
for word1, d in two_grams.items():
    for word2, count in d.items():
        print(f'{word1},{word2},{count}')
