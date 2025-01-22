%formhe-problem:B
%formhe-timestamp:1
%formhe-selfeval-lines:3
%formhe-selfeval-fix:s(X) :- e(X, E).

inter(X, Y) :- e(X, E), e(Y, E), X != Y.
 :- inter(X, Y), sel(X), sel(Y).
k { sel(X) : s(X) } k.
s(X) :- inter(X, Y).