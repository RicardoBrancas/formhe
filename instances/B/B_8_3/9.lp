%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:inter(X, Y) :- e(X, E), e(Y, E), X != Y.  :- inter(X, Y), sel(X), sel(Y). k { sel(X) : s(X) } k.

s(X) :- e(X, E).
inter(X, Y) :- e(X, Y), e(E, E), X != Y.
 :- inter(Y, Y), sel(X), sel((0..X)).
k { sel(X) : s(X) } k :- inter((1..0); (0..1), 0; 0; (0..1)).