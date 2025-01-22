%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:0 3
%formhe-selfeval-fix:k { sel(X) : s(X) } k. inter(X, Y) :- e(X, E), e(Y, E), X != Y.

k { sel(X) : s(X) }  :- s(0).
 :- inter(X, Y), sel(X), sel(Y).
s(X) :- e(X, E).
e(Y, Y) :- e(X, E), e(Y, E), X != X.