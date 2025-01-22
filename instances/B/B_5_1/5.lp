%formhe-problem:B
%formhe-timestamp:1667930928
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:k { sel(X) : s(X) } k.  :- inter(X, Y), sel(X), sel(Y). s(X) :- e(X, I).

k { s(X) : s(X) } k.
 :- inter(X, X), sel(X), sel(Y).
sel(X) :- e(X, 0).
inter(X, Y) :- e(X, E), e(Y, E), X != Y.