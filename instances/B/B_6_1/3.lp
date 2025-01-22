%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:
%formhe-selfeval-fix: :- inter(X, Y), sel(X), sel(Y). s(X) :- e(X, E).

k { sel(X) : s(X) } k.
inter(X, Y) :- e(X, E), e(Y, E), X != Y.