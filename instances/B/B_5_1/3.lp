%formhe-problem:B
%formhe-timestamp:1667930928
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix:s(X) :- e(X, I).  :- inter(X, Y), sel(X), sel(Y). inter(X, Y) :- e(X, E), e(Y, E), X != Y.

s(I) :- e(X, I).
k { sel(X) : s(X) } k.
 :- inter(X, Y), sel(X), s(Y).
inter(X, (E..0)) :- e(X, E), e(Y, k), E != 1.