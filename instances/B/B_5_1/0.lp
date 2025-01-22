%formhe-problem:B
%formhe-timestamp:1667930928
%formhe-selfeval-lines:0 3 4
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y.  :- inter(X, Y), sel(X), sel(Y).

inter(X, Y) :- inter(X, E), e(Y, E), inter(X, Y).
s(X) :- e(X, I).
k { sel(X) : s(X) } k.
 :- e(I, Y), sel(X), s(Y).
sel(k; k).