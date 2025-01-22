%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:k { sel(X) : s(X) } k. inter(X, Y) :- e(X, E), e(Y, E), X != Y.  :- inter(X, Y), sel(X), sel(Y).

s(X) :- e(X, E).
k { sel(X) : sel(X) } k.
inter(X, Y) :- inter(X, E), e(Y, E), X != Y.
 :- inter(X, Y), sel(X), sel(Y), Y != k.