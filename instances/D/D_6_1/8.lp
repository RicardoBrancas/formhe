%formhe-problem:D
%formhe-timestamp:1668079125
%formhe-selfeval-lines:2
%formhe-selfeval-fix: { sel(X) : node(X) } k.

node(X) :- e(_, X).
 :- node(X), node(Y), X != Y, e(X, Y), not sel(X), not sel(Y).
 { node(X) : node(X) } k.
node(X) :- e(X, _).