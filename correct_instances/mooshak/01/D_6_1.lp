%formhe-problem:D
%formhe-timestamp:1668079126

node(X) :- e(X,_).
node(X) :- e(_, X).

{ sel(X) : node(X) } k.

:- node(X), node(Y), X != Y, e(X,Y), not sel(X), not sel(Y).