%formhe-problem:B
%formhe-timestamp:1667929597
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix:set(X) :- e(X, E).  :- joint(X, Y), set(X), set(Y), sel(X), sel(Y). joint(X, Y) :- e(X, E), e(Y, E), X != Y.

set(X) :- joint(X, I0).
 :- joint(X, Y), set(X), set(_), sel(X), set(Y).
k { sel(X) : set(X) } k.
joint(E, Y) :- e(X, E), e(Y, E), X != Y.