%formhe-problem:D
%formhe-timestamp:1698339288
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix: :- e(X, Y), not sel(X), not sel(Y). vertex(V) :- e(_, V).

 :- e(X, V), not sel(X), -sel(Y).
vertex(V) :- e(V, _).
 :- e(_, V).
1 { sel(V) : vertex(V) } k.