%formhe-problem:D
%formhe-timestamp:1668079125
%formhe-selfeval-lines:
%formhe-selfeval-fix:node(X) :- e(X, _).  { sel(X) : node(X) } k.  :- node(X), node(Y), X != Y, e(X, Y), not sel(X), not sel(Y).

node(X) :- e(_, X).