%formhe-groundtruth:../../examples/locking.lp
formhe_definition_begin.

adj(8, 9). adj(8, 4). adj(9, 4). adj(9, 10). adj(4, 10).
adj(4, 2). adj(4, 3). adj(2, 1). adj(3, 1). adj(3, 6).
adj(3, 5). adj(6, 10). adj(6, 5). adj(12, 10). adj(11, 10).
adj(11, 7). adj(6, 7). adj(5, 7).

exit(5).

formhe_definition_end.

room(R1) :- adj(R1, R2).
room(R2) :- adj(R1, R2).

reach(R, R) :- room(R).

reach(R1, R2) :- room(R1), room(R2), room(R3), reach(R1, R3), evac(R3, R2).

ok(R) :- room(R), reach(R, X), exit(X).

%:- not ok(R), room(R).

{ evac(R1, R2) ; evac(R2, R1) } 1 :- adj(R1, R2).

#minimize{ 1, R1, R2 : evac(R1, R2) }.