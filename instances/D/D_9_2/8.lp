%formhe-problem:D
%formhe-timestamp:1698339288
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix:vertex(V) :- e(_, V).  :- e(X, Y), not sel(X), not sel(Y).

vertex(V) :- e(V, _).
sel(V) :- e(_, V).
1 { sel(V) : vertex(V) } k.
 :- e(X, Y), not vertex(X), not sel(Y).