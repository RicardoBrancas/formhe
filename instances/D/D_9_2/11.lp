%formhe-problem:D
%formhe-timestamp:1698339288
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:vertex(V) :- e(V, _). 1 { sel(V) : vertex(V) } k.

 :- e(V, _).
vertex(V) :- e(_, V).
0 { sel(V) : vertex(V) } k.
 :- e(X, Y), not sel(X), not sel(Y).