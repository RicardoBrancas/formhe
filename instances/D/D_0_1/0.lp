%formhe-problem:D
%formhe-timestamp:1
%formhe-selfeval-lines:3
%formhe-selfeval-fix:0 { sel(X) : vx(X) } k.

vx(X) :- e(Y, X).
vx(X) :- e(X, Y).
 :- not sel(X), not sel(Y), e(X, Y).
0 { sel(X) : sel(X) } k.