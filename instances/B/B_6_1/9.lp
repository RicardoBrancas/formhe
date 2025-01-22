%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:2
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y.

k { sel(X) : s(X) } k.
 :- inter(X, Y), sel(X), sel(Y).
inter(X, Y) :- e(X, E), e(Y, E), X == Y.
s(X) :- e(X, E).