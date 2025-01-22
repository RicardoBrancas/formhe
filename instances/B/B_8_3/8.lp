%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:1
%formhe-selfeval-fix:k { sel(X) : s(X) } k.

s(X) :- e(X, E).
k { s(X) : s(X) } k.
inter(X, Y) :- e(X, E), e(Y, E), X != Y.
 :- inter(X, Y), sel(X), sel(Y).