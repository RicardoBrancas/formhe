%formhe-problem:B
%formhe-timestamp:1
%formhe-selfeval-lines:3 4
%formhe-selfeval-fix: :- inter(X, Y), sel(X), sel(Y).

inter(X, Y) :- e(X, E), e(Y, E), X != Y.
s(X) :- e(X, E).
k { sel(X) : s(X) } k.
 :- X != Y, sel(X), sel(Y).
 { sel(X) : sel(X) } 0.