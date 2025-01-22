%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y. k { sel(X) : s(X) } k.

s(X) :- e(X, E).
X != Y :- e(X, E), e(Y, E), X != Y.
 :- inter(X, Y), sel(X), sel(Y).
1 { sel(X) : s(X) } 1.