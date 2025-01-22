%formhe-problem:B
%formhe-timestamp:1667929597
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:joint(X, Y) :- e(X, E), e(Y, E), X != Y. set(X) :- e(X, E).  :- joint(X, Y), set(X), set(Y), sel(X), sel(Y).

k { sel(X) : set(X) } k.
joint(X, Y) :- e(X, E), e(Y, E), k != Y.
set(X) :- e(X, k).
 :- joint(X, E), set(X), set(Y), sel(X), sel(Y).