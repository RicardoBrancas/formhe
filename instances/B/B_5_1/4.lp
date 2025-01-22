%formhe-problem:B
%formhe-timestamp:1667930928
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:s(X) :- e(X, I). inter(X, Y) :- e(X, E), e(Y, E), X != Y.

k { sel(X) : s(X) } k.
 :- inter(X, Y), sel(X), sel(Y).
sel(X) :- e(X, _).
e(X, Y) :- e(X, E), e(Y, I0), inter(X, Y).