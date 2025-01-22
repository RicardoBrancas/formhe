%formhe-problem:D
%formhe-timestamp:1698339288
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- e(X, Y), not sel(X), not sel(Y).

Y != V :- e(X, Y), not sel(X), not vertex(Y).
vertex(V) :- e(V, _).
vertex(V) :- e(_, V).
1 { sel(V) : vertex(V) } k.