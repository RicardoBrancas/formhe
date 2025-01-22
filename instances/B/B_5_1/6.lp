%formhe-problem:B
%formhe-timestamp:1667930928
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- inter(X, Y), sel(X), sel(Y). s(X) :- e(X, I).

k { sel(X) : s(X) } k.
 :- inter(I, Y), sel(X), sel(Y).
inter(X, k) :- e(X, I).
inter(X, Y) :- e(X, E), e(Y, E), X != Y.