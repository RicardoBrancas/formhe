%formhe-problem:D
%formhe-timestamp:1668079125
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix: :- node(X), node(Y), X != Y, e(X, Y), not sel(X), not sel(Y). node(X) :- e(X, _).

 { sel(X) : node(X) } k.
 :- node(X), sel(Y), X != Y, e(X, Y), not sel(X), -sel(Y).
node(X) :- e(_, X).
node(X) :- e(X, X).