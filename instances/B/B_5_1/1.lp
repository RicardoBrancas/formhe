%formhe-problem:B
%formhe-timestamp:1667930928
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y.  :- inter(X, Y), sel(X), sel(Y).

s(X) :- e(X, I).
k { sel(X) : s(X) } k.
e(X, Y) :- X == 0, e(Y, E), X != k.
 :- inter(X, Y), inter(I, 1; I1), sel(_).