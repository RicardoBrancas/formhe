Answer Set Programming (ASP)
============================

.. contents::
    :local:
    :backlinks: none

ASP Program Completer: ``asp_completer.py``
-------------------------------------------

FormHe allows users to find completions for ASP programs with missing parts. Currently, the only supported supported completions are of the Test kind. Furthermore, the tool does not yet support interacting with the user and requires a correct and complete program to be supplied for evaluation. This is done through a special directive on the ``.lp`` file:

``%formhe-groundtruth: ./relative/path/to/complete_program.lp``

Furthermore, ``asp_completer`` does not yet support using predicates declared in the ASP program. It only supports operations supported by the Z3 SMT solver.

How Does it Work?
^^^^^^^^^^^^^^^^^

As explained before, ``asp_completer`` makes use of two ASP programs: an incomplete program, *I*, for which we want to find a missing statement, and a complete program, *C*. The ``asp_completer`` starts by finding models of *I* that are not models of *C* -- essentially incorrect solutions that need to be blocked by the missing statement. This is done by generating models of *I*, adding those models as assumptions to *C* and checking if it is still satisfiable or not. Importanto note: we also use clingo to obtain an UNSAT core -- this allows us to discard unimportant parts of the models focusing on where the problems are.

Now from those UNSAT cores which satisfy *I* but not *C*  we choose the one with the fewest elements. We do this because the smallest UNSAT core has the best chance of generalizing to the other ones. For this smallest core, we record which atoms are in the core, as well as the arguments for those atoms. Consider the following example UNSAT core: ``['queen(1,1)', 'queen(2,2)']``. For this core we would record the following variables ``(queen_A_0 Int) (queen_A_1 Int) (queen_B_0 Int) (queen_B_1 Int)``.


The last step involves program synthesis, in particular SyGuS (through CVC5). Our goal is to synthesize a Boolean function which takes as input the variables we selected in the previous step. This synthesized function should return ``True`` for all the UNSAT cores. Additionally, in order to refine the synthesis problem, we can also use some models from *C*, for which the function should always return ``False``. This allows us to construct a Test statement to add to the ASP program. This statement will block all the found models of *I* that are not models of *C*, while not affecting any of the models of *C*.

Command Line Usage
^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: formhe.asp_completer
   :func: mk_argument_parser
   :prog: asp_completer

Example
^^^^^^^

Consider the following ASP program for solving the n-Queens problem


.. literalinclude :: /../generated_instances/nqueens/no_diag.lp
    :language: prolog

Notice that two of the Test instructions of this program are commented. These are the instructions we wish ``asp_completer`` to write for us. Running ``asp_completer`` over this example produces the following output::

    $ python asp_completer.py example.lp

    Computing models: 100%|██████████| 2000/2000 [00:06<00:00, 300.00it/s]
    Computing gt models:   5%|▍         | 92/2000 [00:00<00:09, 209.08it/s]
    Computing models: 100%|██████████| 2/2 [00:00<00:00, 938.53it/s]
    Using the following UNSAT cores as positive examples:
    ['queen(1,1)', 'queen(2,2)']
    ['queen(1,1)', 'queen(4,2)', 'queen(5,3)']
    ['queen(1,1)', 'queen(4,3)', 'queen(5,2)']
    ['queen(1,1)', 'queen(5,2)', 'queen(6,3)']
    ['queen(1,1)', 'queen(5,3)', 'queen(6,2)']
    ['queen(1,1)', 'queen(6,2)', 'queen(7,3)']
    ['queen(1,1)', 'queen(6,3)', 'queen(7,2)']
    ['queen(1,1)', 'queen(3,3)', 'queen(5,2)']
    ['queen(1,1)', 'queen(3,3)', 'queen(6,2)']
    ['queen(1,1)', 'queen(3,3)', 'queen(7,2)']

    Using the following valid models as negative examples:
    [row(1), row(2), row(3), row(4), row(5), row(6), row(7), row(8), col(1), col(2), col(3), col(4), col(5), col(6), col(7), col(8), queen(1,1), queen(5,2), queen(8,3), queen(6,4), queen(3,5), queen(7,6), queen(2,7), queen(4,8)]
    [row(1), row(2), row(3), row(4), row(5), row(6), row(7), row(8), col(1), col(2), col(3), col(4), col(5), col(6), col(7), col(8), queen(1,1), queen(6,2), queen(8,3), queen(3,4), queen(7,5), queen(4,6), queen(2,7), queen(5,8)]

    SyGuS Spec:

    (synth-fun f ((queen_A_0 Int) (queen_A_1 Int) (queen_B_0 Int) (queen_B_1 Int)) Bool
        ((S Bool) (B Bool) (I Int) (C constant))
        ((S Bool ((or B S) B))
        (B Bool ((< I I) (> I I) (not B) (and B B) (= C C) (= I I)))
        (I Int (0 1 2 (+ I I) (abs I) (- I I) queen_A_0 queen_A_1 queen_B_0 queen_B_1))
        (C constant (empty))
    )
    (constraint (f 1 1 2 2))
    (constraint (or (f 1 1 4 2) (f 1 1 5 3) (f 4 2 5 3)))
    (constraint (or (f 1 1 4 3) (f 1 1 5 2) (f 4 3 5 2)))
    (constraint (or (f 1 1 5 2) (f 1 1 6 3) (f 5 2 6 3)))
    (constraint (or (f 1 1 5 3) (f 1 1 6 2) (f 5 3 6 2)))
    (constraint (or (f 1 1 6 2) (f 1 1 7 3) (f 6 2 7 3)))
    (constraint (or (f 1 1 6 3) (f 1 1 7 2) (f 6 3 7 2)))
    (constraint (or (f 1 1 3 3) (f 1 1 5 2) (f 3 3 5 2)))
    (constraint (or (f 1 1 3 3) (f 1 1 6 2) (f 3 3 6 2)))
    (constraint (or (f 1 1 3 3) (f 1 1 7 2) (f 3 3 7 2)))
    (constraint (not (f 1 1 5 2)))
    (constraint (not (f 1 1 8 3)))
    (constraint (not (f 1 1 6 4)))
    (constraint (not (f 1 1 3 5)))
    (constraint (not (f 1 1 7 6)))
    (constraint (not (f 1 1 2 7)))
    (constraint (not (f 1 1 4 8)))
    (constraint (not (f 5 2 8 3)))
    (constraint (not (f 5 2 6 4)))
    (constraint (not (f 5 2 3 5)))
    (constraint (not (f 5 2 7 6)))
    (constraint (not (f 5 2 2 7)))
    (constraint (not (f 5 2 4 8)))
    (constraint (not (f 8 3 6 4)))
    (constraint (not (f 8 3 3 5)))
    (constraint (not (f 8 3 7 6)))
    (constraint (not (f 8 3 2 7)))
    (constraint (not (f 8 3 4 8)))
    (constraint (not (f 6 4 3 5)))
    (constraint (not (f 6 4 7 6)))
    (constraint (not (f 6 4 2 7)))
    (constraint (not (f 6 4 4 8)))
    (constraint (not (f 3 5 7 6)))
    (constraint (not (f 3 5 2 7)))
    (constraint (not (f 3 5 4 8)))
    (constraint (not (f 7 6 2 7)))
    (constraint (not (f 7 6 4 8)))
    (constraint (not (f 2 7 4 8)))
    (constraint (not (f 1 1 6 2)))
    (constraint (not (f 1 1 8 3)))
    (constraint (not (f 1 1 3 4)))
    (constraint (not (f 1 1 7 5)))
    (constraint (not (f 1 1 4 6)))
    (constraint (not (f 1 1 2 7)))
    (constraint (not (f 1 1 5 8)))
    (constraint (not (f 6 2 8 3)))
    (constraint (not (f 6 2 3 4)))
    (constraint (not (f 6 2 7 5)))
    (constraint (not (f 6 2 4 6)))
    (constraint (not (f 6 2 2 7)))
    (constraint (not (f 6 2 5 8)))
    (constraint (not (f 8 3 3 4)))
    (constraint (not (f 8 3 7 5)))
    (constraint (not (f 8 3 4 6)))
    (constraint (not (f 8 3 2 7)))
    (constraint (not (f 8 3 5 8)))
    (constraint (not (f 3 4 7 5)))
    (constraint (not (f 3 4 4 6)))
    (constraint (not (f 3 4 2 7)))
    (constraint (not (f 3 4 5 8)))
    (constraint (not (f 7 5 4 6)))
    (constraint (not (f 7 5 2 7)))
    (constraint (not (f 7 5 5 8)))
    (constraint (not (f 4 6 2 7)))
    (constraint (not (f 4 6 5 8)))
    (constraint (not (f 2 7 5 8)))


    Solutions:
    (= queen_A_0 (- queen_B_0 (abs (- queen_A_1 queen_B_1))))

We can see that the solution provided is equivalent to the combination of both lines commented in the ``.lp`` file.

ASP Bug Finding (WIP)
---------------------

ASP bug finding essentially equates to computing MCSs. The user provides a buggy ASP program, *B*. They also provide an atom, or series of atoms, which should appear in some model but don't appear in any. The goal is to try to remove statements from *B* until that atom can be part of some model. Since there might many different combinations of statements that work, our goal will be to find the smallest number of statements that need to be changed.

