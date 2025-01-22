%formhe-problem:D
%formhe-timestamp:1698339288
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- e(X, Y), not sel(X), not sel(Y). 1 { sel(V) : vertex(V) } k. vertex(V) :- e(V, _).

vertex(V) :- e(_, V).
 :- e(X, Y), not sel(Y), not e(Y, 0).
vertex(k) :- e(X, 0).