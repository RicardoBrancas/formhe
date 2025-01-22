%formhe-problem:B
%formhe-timestamp:1
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- inter(X, Y), sel(X), sel(Y). inter(X, Y) :- e(X, E), e(Y, E), X != Y. k { sel(X) : s(X) } k.

s(X) :- e(X, E).
 :- inter(X, Y), sel(X), sel(E), E == I1.
e(X, Y) :- e(X, E), e(X, Y), X != Y.