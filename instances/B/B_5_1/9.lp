%formhe-problem:B
%formhe-timestamp:1667930928
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y. k { sel(X) : s(X) } k.  :- inter(X, Y), sel(X), sel(Y).

inter(X, Y) :- e(X, E), inter(Y, E), e(X, Y).
s(X) :- e(X, I).
k { sel(X) : s(X) } .
 :- e(I1, Y), sel(X), sel(Y).