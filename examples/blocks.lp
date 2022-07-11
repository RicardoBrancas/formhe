#include <incmode>.

formhe_definition_begin.

init(3, 2). init(6, 5). init(9, 8).
init(2, 1). init(5, 4). init(8, 7).
init(1, 0). init(4, 0). init(7, 0).
table(0).
goal(8, 6).
goal(6, 4). goal(5, 7).
goal(4, 2). goal(7, 3).
goal(2, 1). goal(3, 9).
goal(1, 0). goal(9, 0).

formhe_definition_end.

#program base.

do(X, Z) :- init(X, Y), not table(Y), table(Z).
do(X, Y) :- goal(X, Y), not table(Y).
on(X, Y, 0) :- init(X, Y).

#program check(t).

:- query(t), goal(X, Y), not on(X, Y, t).

#program step(t).

{ move(X, Y, t) : do(X, Y) } = 1.

move(X, t) :- move(X, Y, t).
on(X, Y, t) :- move(X, Y, t).
on(X, Y, t) :- one(X, Y, t-1), not move(X, t).
lock(Y, t) :- on(X, Y, t-1), not table(Y).
firm(X, t) :- on(X, Y, t), table(Y).
firm(X, t) :- on(X, Y, t), firm(Y, t).

:- lock(X, t), move(X, t).
:- lock(Y, t), move(X, Y, t).
:- init(Y, Z), #count { X : on(X, Y, t) } > 1.
:- init(Y, Z), #count { Y : on(X, Y, t) } > 1.
:- init(X, Z), not firm(X, t).


