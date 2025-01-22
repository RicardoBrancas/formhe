%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:k { sel(X) : s(X) } k. inter(X, Y) :- e(X, E), e(Y, E), X != Y.  :- inter(X, Y), sel(X), sel(Y).

k { s(X) : s(X) } k.
inter(X, Y) :- e(X, E), e(Y, E), X != Y, inter(1, k).
 :- inter(X, Y), s(X), sel(Y).
s(X) :- e(X, E).