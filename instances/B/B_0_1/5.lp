%formhe-problem:B
%formhe-timestamp:1
%formhe-selfeval-lines:2
%formhe-selfeval-fix:k { sel(X) : s(X) } k.

inter(X, Y) :- e(X, E), e(Y, E), X != Y.
 :- inter(X, Y), sel(X), sel(Y).
k { sel(X) : sel(X) } k.
s(X) :- e(X, E).