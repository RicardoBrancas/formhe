%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y.  :- inter(X, Y), sel(X), sel(Y). k { sel(X) : s(X) } k.

s(X) :- e(X, E).
X != Y :- e(X, Y), e(E, E), X != Y.
 :- inter(X, Y), sel(E), sel((1..E)).
k { sel(X) : s(X) } k :- s((k..1); (0..1)).