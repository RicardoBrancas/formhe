%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:1
%formhe-selfeval-fix:k { sel(X) : s(X) } k.

s(X) :- e(X, E).
 :- inter(1, I1; k; Y).
inter(X, Y) :- e(X, E), e(Y, E), X != Y.
 :- inter(X, Y), sel(X), sel(Y).