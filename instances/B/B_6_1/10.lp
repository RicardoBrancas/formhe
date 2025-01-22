%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:0 2 4
%formhe-selfeval-fix: :- inter(X, Y), sel(X), sel(Y). inter(X, Y) :- e(X, E), e(Y, E), X != Y.

 :- inter(X, Y), sel(k), sel(Y).
k { sel(X) : s(X) } k.
inter(X, Y) :- e(X, Y), inter(E, E), 0 != Y.
s(X) :- e(X, E).
e((E..0), (1..0)).