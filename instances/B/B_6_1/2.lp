%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:0
%formhe-selfeval-fix:s(X) :- e(X, E).

s(X) :- inter(X, E).
k { sel(X) : s(X) } k.
inter(X, Y) :- e(X, E), e(Y, E), X != Y.
 :- inter(X, Y), sel(X), sel(Y).