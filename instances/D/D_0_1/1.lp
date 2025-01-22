%formhe-problem:D
%formhe-timestamp:1
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix: :- not sel(X), not sel(Y), e(X, Y). 0 { sel(X) : vx(X) } k. vx(X) :- e(X, Y).

vx(X) :- e(Y, X).
 :- not sel(0), not sel(Y), e(X, Y).
0 { sel(X) : e(X, (1..0)) } k.
vx(X) :- e(X, I1).