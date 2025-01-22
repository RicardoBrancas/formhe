%formhe-problem:D
%formhe-timestamp:1698344186
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:v(X) :- e(X, _).  :- e(X, Y), not sel(X), not sel(Y).

v(X) :- e(_, X).
v(X) :- e(X, k).
 :- e(X, Y), not sel(X), not v(Y).
 { sel(X) : v(X) } k.