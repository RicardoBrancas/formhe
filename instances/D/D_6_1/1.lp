%formhe-problem:D
%formhe-timestamp:1668079125
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix: { sel(X) : node(X) } k.  :- node(X), node(Y), X != Y, e(X, Y), not sel(X), not sel(Y). node(X) :- e(X, _).

 { node(X) : node(X) } .
 :- node(X), sel(Y), X != Y, X == Y, not sel(Y), -sel(Y).
node(X) :- e(_, X).
node(0) :- e(X, _).