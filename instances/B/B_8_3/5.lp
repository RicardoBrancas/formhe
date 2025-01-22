%formhe-problem:B
%formhe-timestamp:1667565226
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix: :- inter(X, Y), sel(X), sel(Y). s(X) :- e(X, E).

k { sel(X) : s(X) } k.
inter(X, Y) :- e(X, E), e(Y, E), X != Y.
 :- inter(X, Y), s(E), sel(Y).
s(X) :- e(X, I0).