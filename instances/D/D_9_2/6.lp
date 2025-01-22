%formhe-problem:D
%formhe-timestamp:1698339288
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:vertex(V) :- e(V, _).  :- e(X, Y), not sel(X), not sel(Y).

vertex(V) :- e(_, V).
1 { sel(V) : vertex(V) } k.
vertex(V) :- e(V, Y).
 :- e(X, Y), not vertex(X), -sel(I0).