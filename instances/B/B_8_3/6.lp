%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:1
%formhe-selfeval-fix:s(X) :- e(X, E).

k { sel(X) : s(X) } k.
 :- e(X, E).
inter(X, Y) :- e(X, E), e(Y, E), X != Y.
 :- inter(X, Y), sel(X), sel(Y).