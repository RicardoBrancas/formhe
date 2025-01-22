%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y.  :- inter(X, Y), sel(X), sel(Y).

s(X) :- e(X, E).
inter(X, E) :- e(X, E), e(Y, E), X != Y.
 :- inter(X, Y), sel(X), sel(Y), s((0..1)).
k { sel(X) : s(X) } k.