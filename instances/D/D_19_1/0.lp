%formhe-problem:D
%formhe-timestamp:1698344186
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix: { sel(X) : v(X) } k. v(X) :- e(_, X).  :- e(X, Y), not sel(X), not sel(Y).

 { sel(X) : sel(X) } 1.
v(X) :- e(0, X), k == 1.
 :- e(_, Y), -sel(X), not sel(Y), not sel(0).
v(X) :- e(X, _).