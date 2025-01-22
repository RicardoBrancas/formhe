%formhe-problem:B
%formhe-timestamp:1667930928
%formhe-selfeval-lines:
%formhe-selfeval-fix:k { sel(X) : s(X) } k.  :- inter(X, Y), sel(X), sel(Y).

s(X) :- e(X, I).
inter(X, Y) :- e(X, E), e(Y, E), X != Y.