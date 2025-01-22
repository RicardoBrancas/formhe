%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:0
%formhe-selfeval-fix:nodes(X) :- e(X, _).

nodes(X) :- e(X, _), nodes((0..1)).
1 { sel(N) : nodes(N) } k.
 :- e(X, Y), not sel(X), not sel(Y).
nodes(Y) :- e(_, Y).