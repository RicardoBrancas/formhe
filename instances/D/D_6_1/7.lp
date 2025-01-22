%formhe-problem:D
%formhe-timestamp:1668079125
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix: :- node(X), node(Y), X != Y, e(X, Y), not sel(X), not sel(Y). node(X) :- e(_, X).

 :- node(_), node(Y), X != Y, e(X, Y), not sel(X), not sel(Y).
 { sel(X) : node(X) } k.
node(X) :- e(0, X).
node(X) :- e(X, _).