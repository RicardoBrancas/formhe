%formhe-problem:B
%formhe-timestamp:1667929597
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:set(X) :- e(X, E). joint(X, Y) :- e(X, E), e(Y, E), X != Y.  :- joint(X, Y), set(X), set(Y), sel(X), sel(Y).

set(X) :- e(X, Y).
E == Y :- e(X, E), e(Y, E), X != Y.
 :- joint(X, Y), set(X), set(Y), sel(X), set(I0).
k { sel(X) : set(X) } k.