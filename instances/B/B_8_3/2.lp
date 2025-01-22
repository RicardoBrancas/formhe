%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:k { sel(X) : s(X) } k. inter(X, Y) :- e(X, E), e(Y, E), X != Y.  :- inter(X, Y), sel(X), sel(Y).

s(X) :- e(X, E).
k { sel(X) : sel(X) } k.
inter(k, Y) :- e(X, E), e(Y, E), e(Y, Y).
X != Y :- inter(X, Y), sel(X), s(Y).