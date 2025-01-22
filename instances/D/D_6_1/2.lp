%formhe-problem:D
%formhe-timestamp:1668079125
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- node(X), node(Y), X != Y, e(X, Y), not sel(X), not sel(Y).

node(X) :- e(X, _).
node(X) :- e(_, X).
 { sel(X) : node(X) } k.
 :- node(X), Y == I0, I0 != Y, e(X, Y), -sel(X), not sel(X).