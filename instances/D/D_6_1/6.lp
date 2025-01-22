%formhe-problem:D
%formhe-timestamp:1668079125
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:node(X) :- e(_, X).  { sel(X) : node(X) } k.  :- node(X), node(Y), X != Y, e(X, Y), not sel(X), not sel(Y).

node(X) :- e(X, _).
 :- e(_, X).
 { sel(X) : sel(X) } k.
 :- node(X), node(X), X != Y, e(X, Y), not sel(X), not sel(Y).