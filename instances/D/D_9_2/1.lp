%formhe-problem:D
%formhe-timestamp:1698339288
%formhe-selfeval-lines:0 3
%formhe-selfeval-fix:vertex(V) :- e(V, _).  :- e(X, Y), not sel(X), not sel(Y).

vertex(0) :- e(V, _).
vertex(V) :- e(_, V).
1 { sel(V) : vertex(V) } k.
 :- e(X, I1), not sel(X), -sel(I1).