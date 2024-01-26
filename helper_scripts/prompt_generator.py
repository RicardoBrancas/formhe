from pathlib import Path

import glob

prompt_base = """You are given the below ASP program that contains an error.

{program}

Problem description: {description}

Input predicates: {input}

Output predicates: {output}

Instructions: Fix the error on the above program by adding, removing or modifying existing lines.
Do not add comments or code that is not necessary to fix the error.
For your answer, return a set of changelogs, containing one or more fixes to the above program.
The changelogs must be formatted with the below instructions.

Changelog format instructions : Each changelog must start with a description of its included fixes.
The changelog must then list one or more pairs of (OriginalCode, FixedCode) code snippets.
Each OriginalCode snippet must list all consecutive original lines of code that must be replaced, followed by the FixedCode snippet with all consecutive fixed lines of code that must replace the original lines of code.
In each pair, the OriginalCode and FixedCode snippets must start at the same source code line number N.
Each listed code line, in both the OriginalCode and FixedCode snippets, must be prefixed with [N] that matches the line index N in the above snippets.
If new lines are added those code line numbers should appear as [0].

After all the changelogs, you should also indicate the type of modifications included, where the types are <added>, <removed> and <modified>.
Where <added> means new lines were added with the prefix [0], <removed> means lines were removed from the original program and <modified> means lines were modified from the original program.
You should end your reply with a markdown code block containing the full fixed program with no line numbers.

---
ChangeLog: 1
FixDescription: <summary>.
OriginalCode@4-5:
[4] <original code line>
[5] <original code line>
FixedCode@4-5:
[4] <fixed code line>
[5] <fixed code line>
OriginalCode@1-2:
[1] <original code line>
[2] <original code line>
FixedCode@1-1:
[1] <fixed code line>
...
ChangeLog : K
FixDescription: <summary>.
OriginalCode@2-3:
[2] <original code line>
[3] <original code line>
FixedCode@2-3:
[2] <fixed code line>
[3] <fixed code line>
[0] <new code line>

Type: <modified> <removed> <added>

```
full fixed program
```
---
Answer :"""


description_dict = {
    "A": """I want to model the k-coloring problem.
Considering the colors 1..k and an undirected graph given by the edge predicate e(s,t), I want to find a color for each vertex so that no two neighbours are colored by the same color.
There are no isolated nodes.
The program should contain the predicate assign(node, color) that is true iff node is colored by color.""",

    "B": """Consider a set of sets S and a natural positive number k. I want to write an ASP program
that decides whether there are k sets in S that are pair-wise disjoint. Formally,
S′ ⊆ S, |S′| = k, and ∀s1, s2 ∈ S′(s1 ∩ s2 = ∅).
The elements of sets in S are given by the predicate e(set, element). The
program should contain the predicate sel(set) that is true iff set belongs to
the solution S′.""",

    "C": """I want to write an ASP program that decides whether there is a set cover of size k for
some positive natural number k. Formally, given a set of sets S, a set cover is a subset of S′ ⊆ S such that the sets in S
and S′ cover the same set of elements, i.e. ⋃ S = ⋃ S′.
The input is a set of facts of the form e(ELEMENT, SET), indicating that the
element ELEMENT belongs to the set SET. The resulting answer set should contain
the predicate sel(SET), indicating that the set SET is part of the cover.""",

    "D": """Given an undirected graph G = (V, E), a vertex cover is a set of vertices V ′ ⊆ V
such that each edge e ∈ E is incident with at least one vertex in V ′, i.e. if
e = (v1, v2), it must be that v1 ∈ V ′ or v2 ∈ V ′. I want to write an Answer Set program P that finds a vertex cover of limited size.
The program should contain a predicate sel(X), which
is true if and only if the vertex X is part of the vertex cover.
Assume that k is a natural number defining the upper bound on the size of the vertex cover.""",

    "E": """I want to write an ASP encoding to determine if a given graph is bipartite or not.
A graph G = (V, E) is bipartite if the set of vertexes can be
split into two disjoint sets A and B such that V = A ∪ B and there is no edge
(u, v) ∈ E such that both vertexes u and v belong to the same set (either A or
B). The program should contain a predicate set(NODE, GROUP) such that all nodes
must be assigned to either group a or group b. """
}

input_predicates = {
    "A": "e(s,t)",
    "B": "e(set, element)",
    "C": "e(ELEMENT, SET)",
    "D": "e(s,t)",
    "E": "edge(E, A)"
}

output_predicates = {
    "A": "assign(node, color)",
    "B": "sel(set)",
    "C": "sel(SET)",
    "D": "sel(X)",
    "E": "set(NODE, GROUP)"
}



for file in glob.glob('../instances/mooshak/*.lp'):
    path = Path(file)
    basename = path.stem

    if basename.startswith('F'):
        continue

    description = description_dict[basename[0]]
    input_p = input_predicates[basename[0]]
    output_p = output_predicates[basename[0]]

    with open(file) as f:
        lines = [l for l in f.readlines() if not l.startswith("%") and not l == '\n']

    prompt = prompt_base.format(program=''.join(lines), description=description, input=input_p, output=output_p)

    new_file = f'../prompts/{basename}.txt'

    # print(prompt)

    # exit()

    with open(new_file, 'w') as f:
        f.write(prompt)

