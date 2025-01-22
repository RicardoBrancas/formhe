%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix: :- inter(X, Y), sel(X), sel(Y). inter(X, Y) :- e(X, E), e(Y, E), X != Y. s(X) :- e(X, E).

k { sel(X) : s(X) } k.
 :- inter(X, Y), sel(I1), sel(Y).
inter(X, Y) :- inter(_, E), e(Y, E), X == Y.
s(X) :- e(X, Y), 1 != k.