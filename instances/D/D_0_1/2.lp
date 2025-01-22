%formhe-problem:D
%formhe-timestamp:1
%formhe-selfeval-lines:
%formhe-selfeval-fix:0 { sel(X) : vx(X) } k.  :- not sel(X), not sel(Y), e(X, Y).

vx(X) :- e(X, Y).
vx(X) :- e(Y, X).