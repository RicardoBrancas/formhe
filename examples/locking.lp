formhe_definition_begin.

adj(8,9). adj(8,4). adj(9, 4). adj(9, 10). adj(4, 10). adj(4, 2). adj(4, 3). adj(

formhe_definition_end.

room(R1) :- adj(R1, R2).
room(R2) :- adj(R1, R2).