%formhe-problem:B
%formhe-timestamp:1
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y.  :- inter(X, Y), sel(X), sel(Y). s(X) :- e(X, E).

e(X, Y) :- e(X, E), e(Y, E), X != Y.
k { sel(X) : s(X) } k.
 :- inter(X, Y), s(X), sel(Y).
not sel(X) :- e(X, E).