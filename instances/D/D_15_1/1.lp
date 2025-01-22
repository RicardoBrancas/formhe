%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:nodes(X) :- e(X, _). 1 { sel(N) : nodes(N) } k.

nodes(1) :- e(X, _).
1 { sel(N) : nodes(N) } .
 :- e(X, Y), not sel(X), not sel(Y).
nodes(Y) :- e(_, Y).