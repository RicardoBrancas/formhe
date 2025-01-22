%formhe-problem:D
%formhe-timestamp:1
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- not sel(X), not sel(Y), e(X, Y).

vx(X) :- e(Y, X).
0 { sel(X) : vx(X) } k.
vx(X) :- e(X, Y).
 :- not sel(Y), not sel(Y), e(X, Y).