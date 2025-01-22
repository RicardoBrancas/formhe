%formhe-problem:D
%formhe-timestamp:1698339288
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix: :- e(X, Y), not sel(X), not sel(Y). 1 { sel(V) : vertex(V) } k. vertex(V) :- e(V, _).

 :- e(X, Y), not sel(X), not sel(Y), e(X; X, (Y..0)).
vertex(V) :- e(_, V).
sel(V) :- e(V, _).