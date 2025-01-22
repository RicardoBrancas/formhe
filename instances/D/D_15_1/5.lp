%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:nodes(Y) :- e(_, Y).  :- e(X, Y), not sel(X), not sel(Y).

1 { sel(N) : nodes(N) } k.
nodes(X) :- e(X, _).
nodes(Y) :- e(_, Y), e(X, 1).
 :- e(X, Y), not sel(X), not sel(1).