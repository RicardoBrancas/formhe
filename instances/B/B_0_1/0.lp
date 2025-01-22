%formhe-problem:B
%formhe-timestamp:1
%formhe-selfeval-lines:1
%formhe-selfeval-fix:k { sel(X) : s(X) } k.

inter(X, Y) :- e(X, E), e(Y, E), X != Y.
k { s(X) : s(X) } k.
 :- inter(X, Y), sel(X), sel(Y).
s(X) :- e(X, E).