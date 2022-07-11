%%writefile graphs1.lp

% define
edge(1, 2).
edge(2, 3).
edge(2, 5).
edge(3, 4).
edge(5, 4).
edge(6, 7).
edge(6, 8).

connected(X, Y) :- edge(X, Y).
connected(X, Y) :- edge(Y, X).

path(X, Y) :- connected(X, Y).
path(X, Y) :- path(X, Z), path(Z, Y).

:- not path(1, 3).