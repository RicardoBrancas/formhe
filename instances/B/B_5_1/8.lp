%formhe-problem:B
%formhe-timestamp:1667930928
%formhe-selfeval-lines:1
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y.

 :- inter(X, Y), sel(X), sel(Y).
X != Y :- e(X, E), e(Y, E), X != Y.
s(X) :- e(X, I).
k { sel(X) : s(X) } k.