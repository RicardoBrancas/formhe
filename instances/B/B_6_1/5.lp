%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix:k { sel(X) : s(X) } k.  :- inter(X, Y), sel(X), sel(Y). inter(X, Y) :- e(X, E), e(Y, E), X != Y.

 { s(X) : s(X) } k.
 :- inter(X, X), sel(X), s(Y), s(0).
s(X) :- e(X, E).
inter(X, Y) :- e(X, E), e(Y, E), inter(X, _), e(0; 1, (k..0)).