%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:k { sel(X) : s(X) } k.  :- inter(X, Y), sel(X), sel(Y). inter(X, Y) :- e(X, E), e(Y, E), X != Y.

k { s(X) : s(X) } k.
 :- inter(X, Y), sel(X), s(Y), X != 0.
inter(X, Y) :- X != E, e(Y, E), inter(X, Y).
s(X) :- e(X, E).