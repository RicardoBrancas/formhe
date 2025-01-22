%formhe-problem:B
%formhe-timestamp:1668077837
%formhe-selfeval-lines:0 4
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y.

inter(1, Y) :- e(_, _), Y != E, inter(E, Y).
 :- inter(X, Y), sel(X), sel(Y).
k { sel(X) : s(X) } k.
s(X) :- e(X, E).
inter(0, 0).