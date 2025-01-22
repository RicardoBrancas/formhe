%formhe-problem:B
%formhe-timestamp:1667929597
%formhe-selfeval-lines:2
%formhe-selfeval-fix:joint(X, Y) :- e(X, E), e(Y, E), X != Y.

k { sel(X) : set(X) } k.
set(X) :- e(X, E).
joint(X, Y) :- e(X, E), joint(Y, X), X != Y.
 :- joint(X, Y), set(X), set(Y), sel(X), sel(Y).